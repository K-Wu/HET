#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/HGT/HGTBackwardKernels.cu.h"
#include "DGLHackKernel/HGT/HGTForwardExperimental.cu.h"
#include "DGLHackKernel/HGT/HGTForwardKernels.cu.h"
#include "DGLHackKernel/NodeLinear.h"
#include "DGLHackKernel/OpPrototyping/HGTIntermediateData.h"
#include "DGLHackKernel/OpPrototyping/HGTProtoOps.h"

namespace HET {
namespace OpPrototyping {
// TODO: collect input data into a struct; malloc intermediate and output data.

// assume nodes indices are currently sorted according to their node type, or
// even only monotype exists. We use naive for loop outside kernel launch to do
// the linear layer for now.
// TODO: implement more general case where nodes may not be sorted according to
// node type, thus indirection needed
// TODO: optimize the for loop by fusing multiple kernels into one
// work for both k-linear and q-linear
void LinearByNodeType(float *Adev /*weight*/, float *Bdev /*input*/,
                      float *Cdev /*output*/, float *biasDev /*bias*/,
                      int m /*hidden dim*/, int n /*num nodes*/,
                      int k /*hidden dim*/, int num_heads, void *workspace,
                      size_t workspaceSize) {
  // weight, bias, input
  // TODO: check if we can initialize ltHandle once per program run or once per
  // layer
  cublasLtHandle_t ltHandle;
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasLtCreate(&ltHandle));
  for (int idx_head = 0; idx_head < num_heads; idx_head += 1) {
    LtSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
            &Adev[m * n * idx_head], m * num_heads, &Bdev[k * idx_head],
            k * num_heads, &beta, &Cdev[m * idx_head], m * num_heads,
            &biasDev[m * idx_head], workspace, workspaceSize);
  }
  CUBLAS_CHECK(cublasLtDestroy(ltHandle));
}

void KLinearByNodeType(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data) {
  std::cout << "KLinearByNodeType ptrs"
            << (intermediate_data->NodeInputFeatures).Ptr() << " "
            << (intermediate_data->KLinearOutput).Ptr() << " "
            << (weights->KLinearBias).Ptr() << " "
            << (weights->KLinearWeights).Ptr() << std::endl;
  LinearByNodeType((weights->KLinearWeights).Ptr(),
                   (intermediate_data->NodeInputFeatures).Ptr(),
                   (intermediate_data->KLinearOutput).Ptr(),
                   (weights->KLinearBias).Ptr(), hyper_params.klinear_out_dim,
                   hyper_params.num_nodes, hyper_params.input_dim,
                   hyper_params.num_heads,
                   (intermediate_data->ltsgemm_workspace).Ptr(),
                   intermediate_data->ltsgemm_workspaceSize);
}

void QLinearByNodeType(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data) {
  // TODO: check if we need to assign separate workspace to each of the linear
  // layer
  std::cout << "QLinearByNodeType ptrs"
            << (intermediate_data->NodeInputFeatures).Ptr() << " "
            << (intermediate_data->QLinearOutput).Ptr() << " "
            << (weights->QLinearBias).Ptr() << " "
            << (weights->QLinearWeights).Ptr() << std::endl;
  LinearByNodeType((weights->QLinearWeights).Ptr(),
                   (intermediate_data->NodeInputFeatures).Ptr(),
                   (intermediate_data->QLinearOutput).Ptr(),
                   (weights->QLinearBias).Ptr(), hyper_params.qlinear_out_dim,
                   hyper_params.num_nodes, hyper_params.input_dim,
                   hyper_params.num_heads,
                   (intermediate_data->ltsgemm_workspace).Ptr(),
                   intermediate_data->ltsgemm_workspaceSize);
}

void VLinearByNodeType(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data) {
  std::cout << "VLinearByNodeType ptrs"
            << (intermediate_data->NodeInputFeatures).Ptr() << " "
            << (intermediate_data->VLinearOutput).Ptr() << " "
            << (weights->VLinearBias).Ptr() << " "
            << (weights->VLinearWeights).Ptr() << std::endl;
  LinearByNodeType((weights->VLinearWeights).Ptr(),
                   (intermediate_data->NodeInputFeatures).Ptr(),
                   (intermediate_data->VLinearOutput).Ptr(),
                   (weights->VLinearBias).Ptr(), hyper_params.vlinear_out_dim,
                   hyper_params.num_nodes, hyper_params.input_dim,
                   hyper_params.num_heads,
                   (intermediate_data->ltsgemm_workspace).Ptr(),
                   intermediate_data->ltsgemm_workspaceSize);
}

void ALinearByNodeType(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data) {
  // TODO: the input to ALinear should be intermediate data
  std::cout << "ALinearByNodeType ptrs"
            << (intermediate_data->NodeInputFeatures).Ptr() << " "
            << (intermediate_data->ALinearOutput).Ptr() << " "
            << (weights->ALinearBias).Ptr() << " "
            << (weights->ALinearWeights).Ptr() << std::endl;
  LinearByNodeType((weights->ALinearWeights).Ptr(),
                   (intermediate_data->NodeInputFeatures).Ptr(),
                   (intermediate_data->ALinearOutput).Ptr(),
                   (weights->ALinearBias).Ptr(), hyper_params.alinear_out_dim,
                   hyper_params.num_nodes, hyper_params.input_dim,
                   hyper_params.num_heads,
                   (intermediate_data->ltsgemm_workspace).Ptr(),
                   intermediate_data->ltsgemm_workspaceSize);
}
}  // namespace OpPrototyping
}  // namespace HET