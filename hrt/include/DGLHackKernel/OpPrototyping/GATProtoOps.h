#pragma once

#include "DGLHackKernel/DGLHackUtils.h"
#include "DGLHackKernel/GAT/FusedGAT.cu.h"
#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"
#include "MySimpleNDArray/MySimpleNDArray.h"

namespace HET {
namespace OpPrototyping {
// from seastar dgl-hack
template </*int XPU, */ typename Idx, typename DType>
void FusedGatKernelImpl(
    MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>>
        incsr,  // create incsr in the driver logic
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &feat_src,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &el,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &er,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &sum,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &exp,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret, float slope) {
  // As GAT only has 1 type of relationship, we use a specialcase of separateCSR
  // where num releationship is asserted as 1
  assert(incsr.num_rels == 1);
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  // zero out ret, and packing feat_src, el, er, ret, graph together into one
  // struct using raw float pointers get csr matrix
  GatFusedData<Idx, DType> gdata;

  int64_t el_xlen = el.SeastarComputeXLength();
  int64_t feat_src_xlen = feat_src.SeastarComputeXLength();
  int64_t ret_len = ret.SeastarComputeXLength();

  gdata.feat_src = feat_src.Ptr();
  gdata.el = el.Ptr();
  gdata.er = er.Ptr();
  gdata.sum = sum.Ptr();
  gdata.exp = exp.Ptr();
  gdata.ret = ret.Ptr();
  gdata.leaky_relu_slope = slope;
  int64_t num_rows = el.data.size() / el_xlen;
  gdata.num_heads = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;

  gdata.eids = static_cast<Idx *>(thrust::raw_pointer_cast(incsr.eids.data()));

  // write a device function and call it from here

  // Configure kernel launch parameters.
  // TODO: we can safely reshape (nthrs_x, nthrs_y) to assign more y dimension
  // to rows as usually n_head is smaller than 32
  int nthrs_x = 32;
  int nthrs_y = 1;
  int nblks_x = (el_xlen + nthrs_x - 1) / (nthrs_x);
  int nblks_y = std::min(num_rows, (int64_t)MAX_NBLKS);
  const dim3 nblks(nblks_x, nblks_y);
  const dim3 nthrs(nthrs_x, nthrs_y);

  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  HET_gatExpLeakyReluSumKernel<Idx, DType, CompactAsOfNodeKind::Enabled, false>
      <<<nblks, nthrs>>>(
          gdata,
          static_cast<Idx *>(thrust::raw_pointer_cast(incsr.row_ptrs.data())),
          static_cast<Idx *>(
              thrust::raw_pointer_cast(incsr.col_indices.data())),
          {}, incsr.num_rows, {});
  nthrs_x = SeastarFindNumThreads(el_xlen, 64);
  nthrs_y = SeastarFindNumThreads(feat_src_xlen / el_xlen, MAX_NTHRS / nthrs_x);
  nblks_x = 1;
  nblks_y = std::min(num_rows, (int64_t)MAX_NBLKS);
  const dim3 nthrs2(nthrs_x, nthrs_y);
  const dim3 nblks2(nblks_x, nblks_y);

  HET_gatSumProdZipDivKernel<Idx, DType, CompactAsOfNodeKind::Enabled, false>
      <<<nblks2, nthrs2>>>(
          gdata,
          static_cast<Idx *>(thrust::raw_pointer_cast(incsr.row_ptrs.data())),
          static_cast<Idx *>(
              thrust::raw_pointer_cast(incsr.col_indices.data())),
          {}, incsr.num_rows, {});
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "FusedGatKernelImpl fused<" << 0 << "> time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;
}

// from seastar dgl-hack
template </*int XPU, */ typename Idx, typename DType, bool FLAG_KERNEL_FUSED>
void BackwardFusedGatKernelImpl(
    // create CSR in driver code
    MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> outcsr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &feat_src,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &el,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &er,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &sum,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &exp,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &ret,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_out,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_feat_src,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_el,
    MySimpleNDArray<DType, thrust::device_allocator<DType>> &grad_er,
    // TODO: check if it needs a different eid
    float slope) {
  // As GAT only has 1 type of relationship, we use a specialcase of separateCSR
  // where num releationship is asserted as 1
  assert(outcsr.num_rels == 1);
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  // zero out ret, and packing feat_src, el, er, ret, graph together into one
  // struct using raw float pointers get csr matrix
  BackwardGatFusedData<Idx, DType> gdata;
  int64_t el_xlen = el.SeastarComputeXLength();
  int64_t feat_src_xlen = feat_src.SeastarComputeXLength();
  gdata.feat_src = feat_src.Ptr();
  gdata.el = el.Ptr();
  gdata.er = er.Ptr();
  gdata.sum = sum.Ptr();
  gdata.exp = exp.Ptr();
  gdata.ret = ret.Ptr();
  gdata.grad_out = grad_out.Ptr();
  gdata.grad_feat_src = grad_feat_src.Ptr();
  gdata.grad_el = grad_el.Ptr();
  gdata.grad_er = grad_er.Ptr();
  gdata.leaky_relu_slope = slope;
  int num_rows = el.data.size() / el_xlen;
  gdata.num_heads = el_xlen;
  gdata.feat_src_xlen = feat_src_xlen;
  gdata.eids = static_cast<Idx *>(thrust::raw_pointer_cast(outcsr.eids.data()));
  // write a device function and call it from here

  // Configure kernel launch parameters.
  int nthrs_x = SeastarFindNumThreads(el_xlen, 64);
  int nthrs_y =
      SeastarFindNumThreads(feat_src_xlen / el_xlen, MAX_NTHRS / nthrs_x);
  int nblks_x = 1;

  int nblks_y = std::min(num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  if constexpr (!FLAG_KERNEL_FUSED) {
    HET_fusedGatBackwardGradFeatSrc<Idx, DType, CompactAsOfNodeKind::Enabled,
                                    false><<<nblks, nthrs>>>(
        gdata,
        static_cast<Idx *>(thrust::raw_pointer_cast(outcsr.row_ptrs.data())),
        static_cast<Idx *>(thrust::raw_pointer_cast(outcsr.col_indices.data())),
        {}, outcsr.num_rows, {});

    HET_fusedGatBackwardGradElEr<Idx, DType, CompactAsOfNodeKind::Enabled,
                                 false><<<nblks, nthrs>>>(
        gdata,
        static_cast<Idx *>(thrust::raw_pointer_cast(outcsr.row_ptrs.data())),
        static_cast<Idx *>(thrust::raw_pointer_cast(outcsr.col_indices.data())),
        {}, outcsr.num_rows, {});
  } else {
    HET_fusedGatBackwardGradElErFeatSrcFused<
        Idx, DType, CompactAsOfNodeKind::Enabled, false><<<nblks, nthrs>>>(
        gdata,
        static_cast<Idx *>(thrust::raw_pointer_cast(outcsr.row_ptrs.data())),
        static_cast<Idx *>(thrust::raw_pointer_cast(outcsr.col_indices.data())),
        {}, outcsr.num_rows, {});
  }
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "BackwardFusedGatKernelImpl fused<" << FLAG_KERNEL_FUSED << "> time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;
}
}  // namespace OpPrototyping
}  // namespace HET