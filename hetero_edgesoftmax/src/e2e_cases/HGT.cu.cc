#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/OpPrototyping/HGTOps.h"

int main() {
  cusp::csr_matrix<int, int, cusp::host_memory> fb15k237_graph =
      LoadFB15k237Data();
}