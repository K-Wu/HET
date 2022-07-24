#include "DGLHackKernel/DGLHackKernel.h"


int main(){
    //cusp::csr_matrix<int, int, cusp::host_memory> fb15k237_graph = LoadFB15k237Data();
    //FusetGATProfiling_main(fb15k237_graph, 4, 64);
    //RGCNLayer1Profiling_main(fb15k237_graph, 32, 32);
    //RGCNLayer1Profiling_MyHYB_main(fb15k237_graph, 32, 32);
    //RGCNLayer1Profiling_main_check_correctness(fb15k237_graph, 32, 32);

    auto segment_csr = LoadSegmentCSR_OGBN_MAG();
    //try{
    _HGTExperimental_main(segment_csr,4, 64, 64);
    // }
    // catch(std::exception& e)
    // {
    //     std::cout << e.what() << std::endl;
    // }
    // catch(thrust::system::system_error& e)
    // {
    //     std::cout << e.what() << std::endl;
    // }
    // catch(...)
    // {
    //     std::cout << "unknown exception" << std::endl;
    // }
    
    return 0;
}