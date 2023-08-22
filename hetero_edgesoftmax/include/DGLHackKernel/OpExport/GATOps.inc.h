#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "DGLHackKernel/DGLHackUtils.h"
#include "ThreadingGridsBlocksSchedules.h"
namespace HET {
namespace TorchExport {
namespace RGCN {
namespace FwProp {
namespace IntegratedCSR {
template </*int XPU, */ typename Idx, typename DType>
void _FusedKernelImpl(at::Tensor &incsr_row_ptrs, at::Tensor &incsr_col_indices,
                      at::Tensor &incsr_eids, at::Tensor &feat_src,
                      at::Tensor &el, at::Tensor &er, at::Tensor &sum,
                      at::Tensor &exp, at::Tensor &ret, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // As GAT only has 1 type of relationship, we use a specialcase of separateCSR
  // where num releationship is asserted as 1

  GatFusedData<Idx, DType> gdata{.feat_src_xlen =
                                     SeastarComputeXLength(feat_src),
                                 .num_heads = SeastarComputeXLength(el),
                                 .eids = incsr_eids.data_ptr<Idx>(),
                                 .leaky_relu_slope = static_cast<float>(slope),
                                 .feat_src = feat_src.data_ptr<DType>(),
                                 .el = el.data_ptr<DType>(),
                                 .er = er.data_ptr<DType>(),
                                 .sum = sum.data_ptr<DType>(),
                                 .exp = exp.data_ptr<DType>(),
                                 .ret = ret.data_ptr<DType>()};

  // Configure kernel launch parameters.
  // NB: updated to Type 1 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-069c3c2c5a9041df2c9a0b01c9f28044c4d519d86c5ed2f859d0d74282967062L232-R233
  // head -> blockIdx.x * blockDim.x + threadIdx.x;
  // node -> blockIdx.y * blockDim.y + threadIdx.y;
  int64_t incsr_num_rows = incsr_row_ptrs.numel() - 1;
  auto [nblks, nthrs] = get_type1_schedule(gdata.num_heads, incsr_num_rows);

  // it is okay to pass in nullptrs as mapper data because !RelationalFlag
  HET_gatExpLeakyReluSumKernel<Idx, DType, CompactAsOfNodeKind::Enabled, false>
      <<<nblks, nthrs, 0, stream>>>(gdata, incsr_row_ptrs.data_ptr<Idx>(),
                                    incsr_col_indices.data_ptr<Idx>(), {},
                                    incsr_num_rows, {});

  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  auto [nblks2, nthrs2] =
      get_type2_schedule(gdata.num_heads, gdata.feat_src_xlen, incsr_num_rows);
  // it is okay to pass in nullptrs as mapper data because !RelationalFlag
  HET_gatSumProdZipDivKernel<Idx, DType, CompactAsOfNodeKind::Enabled, false>
      <<<nblks2, nthrs2, 0, stream>>>(gdata, incsr_row_ptrs.data_ptr<Idx>(),
                                      incsr_col_indices.data_ptr<Idx>(), {},
                                      incsr_num_rows, {});
}
constexpr auto FusedKernelImpl = _FusedKernelImpl<int64_t, float>;
} // namespace IntegratedCSR
} // namespace FwProp
namespace BckProp {
namespace IntegratedCSR {
template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED>
void _FusedKernelImpl(at::Tensor &outcsr_row_ptrs,
                      at::Tensor &outcsr_col_indices, at::Tensor &outcsr_eids,
                      at::Tensor &feat_src, at::Tensor &el, at::Tensor &er,
                      at::Tensor &sum, at::Tensor &exp, at::Tensor &ret,
                      at::Tensor &grad_out, at::Tensor &grad_feat_src,
                      at::Tensor &grad_el, at::Tensor &grad_er, double slope) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // As GAT only has 1 type of relationship, we use a specialcase of separateCSR
  // where num releationship is asserted as 1
  BackwardGatFusedData<Idx, DType> gdata{
      .feat_src_xlen = SeastarComputeXLength(feat_src),
      .num_heads = SeastarComputeXLength(el),
      .eids = outcsr_eids.data_ptr<Idx>(),
      .leaky_relu_slope = static_cast<float>(slope),
      .feat_src = feat_src.data_ptr<DType>(),
      .el = el.data_ptr<DType>(),
      .er = er.data_ptr<DType>(),
      .sum = sum.data_ptr<DType>(),
      .exp = exp.data_ptr<DType>(),
      .ret = ret.data_ptr<DType>(),
      .grad_out = grad_out.data_ptr<DType>(),
      .grad_feat_src = grad_feat_src.data_ptr<DType>(),
      .grad_el = grad_el.data_ptr<DType>(),
      .grad_er = grad_er.data_ptr<DType>()};

  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // edge|node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  int64_t outcsr_num_rows = outcsr_row_ptrs.numel() - 1;
  auto [nblks, nthrs] =
      get_type2_schedule(gdata.num_heads, gdata.feat_src_xlen, outcsr_num_rows);

  if constexpr (!FLAG_KERNEL_FUSED) {
    HET_fusedGatBackwardGradFeatSrc<Idx, DType, CompactAsOfNodeKind::Enabled,
                                    false><<<nblks, nthrs, 0, stream>>>(
        gdata, outcsr_row_ptrs.data_ptr<Idx>(),
        outcsr_col_indices.data_ptr<Idx>(), {}, outcsr_num_rows, {});
    HET_fusedGatBackwardGradElEr<Idx, DType, CompactAsOfNodeKind::Enabled,
                                 false><<<nblks, nthrs, 0, stream>>>(
        gdata, outcsr_row_ptrs.data_ptr<Idx>(),
        outcsr_col_indices.data_ptr<Idx>(), {}, outcsr_num_rows, {});
  } else {
    // it is okay to pass in nullptrs as mapper data because !RelationalFlag
    HET_fusedGatBackwardGradElErFeatSrcFused<
        Idx, DType, CompactAsOfNodeKind::Enabled, false>
        <<<nblks, nthrs, 0, stream>>>(gdata, outcsr_row_ptrs.data_ptr<Idx>(),
                                      outcsr_col_indices.data_ptr<Idx>(), {},
                                      outcsr_num_rows, {});
  }
}

constexpr auto FusedKernelImpl = _FusedKernelImpl<int64_t, float, true>;
} // namespace IntegratedCSR
} // namespace BckProp
} // namespace RGCN
} // namespace TorchExport
} // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // Fused GAT CSR Declaration
  m.def("fused_gat_csr", RGCN::FwProp::IntegratedCSR::FusedKernelImpl);
  m.def("backward_fused_gat_csr",
        RGCN::BckProp::IntegratedCSR::FusedKernelImpl);
}
