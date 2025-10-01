#include "zendnn.hpp"
   
   int main() {
       zendnn::engine eng(zendnn::engine::kind::cpu, 0);
       
       printf("ZenDNN engine created: %s\n", 
              eng.get_kind() == zendnn::engine::kind::cpu ? "CPU" : "Unknown");
       
       return 0;
   }