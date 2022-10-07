// from
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu

/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cublasLt.h>

#include "utils.h"

/// Sample wrapper executing single precision gemm with cublasLtMatmul, nearly a
/// drop-in replacement for cublasSgemm, with addition of the workspace to
/// support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul
/// descriptor attribute matmul is not using cublas handle's configuration of
/// math mode, here tensor ops are implicitly allowed; to change this configure
/// appropriate attribute in the preference handle
void LtSgemm(cublasLtHandle_t ltHandle, cublasOperation_t transa,
             cublasOperation_t transb, int m, int n, int k,
             const float *alpha, /* host pointer */
             const float *A, int lda, const float *B, int ldb,
             const float *beta, /* host pointer */
             float *C, int ldc, const void *bias, void *workspace,
             size_t workspaceSize) {
  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;
  // from https://github.com/marian-nmt/marian-dev/pull/778/files
  bool do_relu = true;
  cublasLtEpilogue_t epilogue =
      do_relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // create operation desciriptor; see cublasLtMatmulDescAttributes_t for
  // details about defaults; here we just need to set the transforms for A and B
  CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                              CUBLASLT_MATMUL_DESC_EPILOGUE,
                                              &epilogue, sizeof(epilogue)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

  // create matrix descriptors, we are good with the details here so no need to
  // set any extra attributes
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F,
                                          transa == CUBLAS_OP_N ? m : k,
                                          transa == CUBLAS_OP_N ? k : m, lda));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F,
                                          transb == CUBLAS_OP_N ? k : n,
                                          transb == CUBLAS_OP_N ? n : k, ldb));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

  // create preference handle; here we could use extra attributes to disable
  // tensor ops or to make sure algo selected will work with badly aligned A, B,
  // C; here for simplicity we just assume A,B,C are always well aligned (e.g.
  // directly come from cudaMalloc)
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
      sizeof(workspaceSize)));

  // we just need the best available heuristic to try and run matmul. There is
  // no guarantee this will work, e.g. if A is badly aligned, you can request
  // more (e.g. 32) algos and try to run them one by one until something works
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
      &heuristicResult, &returnedResults));

  if (returnedResults == 0) {
    CUBLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  CUBLAS_CHECK(cublasLtMatmul(
      ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C,
      Cdesc, &heuristicResult.algo, workspace, workspaceSize, 0));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
}

int main() {
  int m = 64;
  int k = 64;
  int n = 1000;
  int num_heads = 4;
  float alpha = 1.f, beta = 0.f;
  size_t workspaceSize = 1024 * 1024 * 4;
  // to use this for node k-linear or q-linear. A as weight. n as number of
  // nodes. B as input. C as output. for multi-head attention, set leading
  // dimension of B as |HEAD_NUM|*|HIDDEN_SIZE|, and leading dimension of C as
  // |HEAD_NUM|*|HIDDEN_SIZE|, and offset B, C by |HIDDEN_SIZE|*i, where i is
  // the head index. offset A by |HIDDEN_SIZE|*|HIDDEN_SIZE|*i. offset bias by
  // |HIDDEN_SIZE|*i.

  std::vector<float> Ahost(m * k * num_heads);
  std::vector<float> Bhost(n * k * num_heads);
  std::vector<float> Chost(m * n * num_heads);
  std::vector<float> biasHost(m * num_heads);
  void *workspace;
  float *Adev, *Bdev;
  float *Cdev, *biasDev;
  cublasLtHandle_t ltHandle;

  // initialization
  CUBLAS_CHECK(cublasLtCreate(&ltHandle));
  cuda_err_chk(cudaMalloc(reinterpret_cast<void **>(&Adev),
                          m * k * num_heads * sizeof(float)));
  cuda_err_chk(cudaMalloc(reinterpret_cast<void **>(&Bdev),
                          n * k * num_heads * sizeof(float)));
  cuda_err_chk(
      cudaMalloc(reinterpret_cast<void **>(&Cdev), m * n * sizeof(float)));
  cuda_err_chk(cudaMalloc(reinterpret_cast<void **>(&biasDev),
                          m * num_heads * sizeof(float)));
  cuda_err_chk(cudaMalloc(&workspace, workspaceSize));

  // data generation
  for (int i = 0; i < m * k * num_heads; i++) Ahost[i] = __float2half_rn(i);
  for (int i = 0; i < n * k * num_heads; i++) Bhost[i] = __float2half_rn(i);
  for (int i = 0; i < m * num_heads; i++) biasHost[i] = __float2half_rn(i + 1);

  // copy data to device
  cuda_err_chk(cudaMemcpy(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]),
                          cudaMemcpyHostToDevice));
  cuda_err_chk(cudaMemcpy(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]),
                          cudaMemcpyHostToDevice));
  cuda_err_chk(cudaMemcpy(biasDev, biasHost.data(),
                          biasHost.size() * sizeof(biasHost[0]),
                          cudaMemcpyHostToDevice));

  // run matmul
  LtSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, &Adev[m * n * 0],
          m * num_heads, &Bdev[k * 0], k * num_heads, &beta, &Cdev[m * 0],
          m * num_heads, &biasDev[m * 0], workspace, workspaceSize);
  LtSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, &Adev[m * n * 1],
          m * num_heads, &Bdev[k * 1], k * num_heads, &beta, &Cdev[m * 1],
          m * num_heads, biasDev, workspace, workspaceSize);
  LtSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, &Adev[m * n * 2],
          m * num_heads, &Bdev[k * 2], k * num_heads, &beta, &Cdev[m * 2],
          m * num_heads, biasDev, workspace, workspaceSize);
  LtSgemm(ltHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, &Adev[m * n * 3],
          m * num_heads, &Bdev[k * 3], k * num_heads, &beta, &Cdev[m * 3],
          m * num_heads, biasDev, workspace, workspaceSize);

  // destruction
  CUBLAS_CHECK(cublasLtDestroy(ltHandle));
  cuda_err_chk(cudaFree(Adev));
  cuda_err_chk(cudaFree(Bdev));
  cuda_err_chk(cudaFree(Cdev));
  cuda_err_chk(cudaFree(biasDev));
  cuda_err_chk(cudaFree(workspace));
}