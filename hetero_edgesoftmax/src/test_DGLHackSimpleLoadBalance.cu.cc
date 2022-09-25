#include "DGLHackKernel/DGLHackKernel.h"


int main(){
    cusp::csr_matrix<int, int, cusp::host_memory> fb15k237_graph = LoadFB15k237Data(false, false);
    cusp::csr_matrix<int, int, cusp::host_memory> fb15k237_graph_sorted_by_srcs = LoadFB15k237Data(true, true);

    RGCNLayer1Profiling_main(fb15k237_graph, 32, 32);
    RGCNLayer1Profiling_main(fb15k237_graph_sorted_by_srcs, 32, 32);

    
    return 0;
}