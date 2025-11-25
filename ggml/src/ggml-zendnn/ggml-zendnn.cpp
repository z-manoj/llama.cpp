#include "ggml-zendnn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "zendnn.hpp"

#include <cstring>


struct ggml_backend_zendnn_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
    zendnn::engine eng{zendnn::engine::kind::cpu, 0};
    zendnn::stream stream{eng};
};

template <typename T>
zendnn::memory::data_type get_zendnn_data_type() {
    if constexpr (std::is_same_v<T, float>) {
        return zendnn::memory::data_type::f32;
    } else if constexpr (std::is_same_v<T, ggml_bf16_t>) {
        return zendnn::memory::data_type::bf16;
    } else {
        return zendnn::memory::data_type::undef;
    }
}

/**
 * ZenDNN matmul: computes C = B * A.
 *
 * - A: weights, shape (k, m), column-major (each column is a weight vector for one output).
 * - B: input, shape (n, k), row-major (each row is an input sample).
 * - C: output, shape (n, m), row-major.
 *
 * Dimensions:
 *   m = output features (columns of C, columns of A)
 *   n = batch size      (rows of C, rows of B)
 *   k = inner dimension (columns of B, rows of A)
 */
template <typename TA, typename TB, typename TC>
static bool ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx,
                               int64_t m, int64_t n, int64_t k, 
                               const TA * A, const TB * B, TC * C) {

    try {
        const zendnn::memory::dims src_dims    = {n, k};  // B: (n,k)
        const zendnn::memory::dims weight_dims = {k, m};  // A: (k,m)
        const zendnn::memory::dims dst_dims    = {n, m};  // C: (n,m)
        
        zendnn::memory::desc src_md(src_dims, get_zendnn_data_type<TB>(), zendnn::memory::format_tag::ab);
        zendnn::memory::desc weight_md(weight_dims, get_zendnn_data_type<TA>(), zendnn::memory::format_tag::ba);
        zendnn::memory::desc dst_md(dst_dims, get_zendnn_data_type<TC>(), zendnn::memory::format_tag::ab);

        zendnn::memory src_mem(src_md, ctx->eng, const_cast<TB *>(B));
        zendnn::memory weight_mem(weight_md, ctx->eng, const_cast<TA *>(A));
        zendnn::memory dst_mem(dst_md, ctx->eng, C);

        zendnn::matmul::desc matmul_d(src_md, weight_md, dst_md);
        zendnn::matmul::primitive_desc matmul_pd(matmul_d, ctx->eng);
        zendnn::matmul matmul_prim(matmul_pd);

        matmul_prim.execute(ctx->stream, {
            {ZENDNN_ARG_SRC, src_mem},
            {ZENDNN_ARG_WEIGHTS, weight_mem},
            {ZENDNN_ARG_DST, dst_mem}
        });

        ctx->stream.wait();
        return true;
    } catch (const std::exception &e) {
        GGML_LOG_ERROR("ZenDNN matmul failed: %s\n", e.what());
        return false;
    }
}

static bool ggml_zendnn_sgemm(ggml_backend_zendnn_context * ctx,
                              int64_t m, int64_t n, int64_t k,
                              const void * A, const void * B, void * C,
                              int Atype, int Btype, int Ctype) {
    
    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);

    // categorize types
    switch (Atype) {
        case GGML_TYPE_F32:
            if (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                return false;
            return ggml_zendnn_matmul<float, float, float>(
                ctx, m, n, k, 
                (const float *)A,
                (const float *)B,
                (float *)C);
        case GGML_TYPE_BF16:
            if (Btype != GGML_TYPE_BF16)
                return false;
            if (Ctype == GGML_TYPE_BF16)
                return ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k, 
                    (const ggml_bf16_t *)A,
                    (const ggml_bf16_t *)B,
                    (ggml_bf16_t *)C);
            if (Ctype == GGML_TYPE_F32)
                return ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, float>(
                    ctx, m, n, k, 
                    (const ggml_bf16_t *)A,
                    (const ggml_bf16_t *)B,
                    (float *)C);
            return false;
        default:
            return false; // unsupported type
    }
}

static void ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_BINARY_OP_LOCALS

    enum ggml_type    const vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t const from_float = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    void * work_data = ctx->work_data.get();
    if (src1->type != vec_dot_type) {
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        const size_t desired_wsize = ne13 * nbw3;
        if (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(new char[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();
        
        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel for collapse(3) num_threads(ctx->n_threads) schedule(static)
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    const float * src1_f32 = (float *)((char *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    void * src1_conv = (char *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const void* wdata = src1->type == vec_dot_type ? src1->data : work_data;
            const size_t row_size = ggml_row_size(vec_dot_type, ne10);
            if (!ggml_zendnn_sgemm(ctx,
                                  ne01,       // m
                                  ne11,       // n
                                  ne10,       // k
                                  static_cast<const char *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  static_cast<const char *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  static_cast<char *>(dst->data) + i12*nb2 + i13*nb3,
                                  src0->type,
                                  vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN sgemm failed\n", __func__);
        }
    }
}

// backend interface

static const char * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    return "ZenDNN";

    GGML_UNUSED(backend);
}

static void ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_zendnn_guid(void) {
    static const char * guid_str = "AMD-ZENDNN-ACCELER";
    return reinterpret_cast<ggml_guid_t>(const_cast<char*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(void) {
    ggml_backend_zendnn_context * ctx = new ggml_backend_zendnn_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_zendnn(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

void ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, int n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
static const char * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    return "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
static const char * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    return "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

static void ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false
    };
}

static ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    if (backend == NULL) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        return NULL;
    }

    return backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const ggml_tensor * weights = op->src[0];
            const ggml_tensor * inputs = op->src[1];

            const int64_t ne10 = inputs->ne[0];
            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            const int64_t min_batch = 1;
            if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                    return false;
            }
            switch (weights->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_BF16:
                    return true;
                default:
                    return false;
            }
        } break;
        
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ NULL,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ NULL,
    /* .event_new              = */ NULL,
    /* .event_free             = */ NULL,
    /* .event_synchronize      = */ NULL,
};

// backend reg interface
static const char * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    return "ZenDNN";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_zendnn_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_zendnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_zendnn_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_count,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(void) {
    static struct ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
