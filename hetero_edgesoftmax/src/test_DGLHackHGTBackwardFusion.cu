#include "DGLHackKernel/DGLHackKernel.h"


int main(){
    auto csr_h = LoadOGBN_MAG();
    
    printf("num_rows %d num_rels %d, total_num_nnzs %d",csr_h.num_rows, csr_h.num_rels, csr_h.total_num_nnzs);
HGTBackPropGradientSMAFusionProfiling_main(csr_h, 4, 64);
}