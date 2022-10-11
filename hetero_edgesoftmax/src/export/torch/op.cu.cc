// from
// https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>
//#include "DGLHackKernel/DGLHackKernel.h"
//#include "hetero_edgesoftmax.h"

at::Tensor dummy_warp_perspective(at::Tensor image, at::Tensor warp) {
  return image.clone();
}

TORCH_LIBRARY(torch_hetero_edgesoftmax, m) {
  m.def("dummy_warp_perspective", dummy_warp_perspective);
}