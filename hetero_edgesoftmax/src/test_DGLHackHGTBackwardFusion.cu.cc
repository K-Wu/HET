#include "DGLHackKernel/DGLHackKernel.h"

int main() {
  auto csr_h = LoadOGBN_MAG();

  printf("num_rows %ld num_rels %ld, total_num_nnzs %ld", csr_h.num_rows,
         csr_h.num_rels, csr_h.total_num_nnzs);
  HGTBackPropGradientSMAFusionProfiling_main(csr_h, 1, 32);
}