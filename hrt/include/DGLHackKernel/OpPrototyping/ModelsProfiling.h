#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include "GATProtoOps.h"
#include "HGTProtoOps.h"
#include "MyHyb/MyHyb.h"
#include "MySimpleNDArray/MySimpleNDArray.h"
#include "RGCNProtoOps.h"

namespace HET {
namespace OpPrototyping {

// TODO: update relative path since switch from make to cmake. Search for
// npy::LoadArrayFromNumpy() invocations. 1/3 in kernel.cu.cc, test_hypb.cu.cc,
// DGLHackKernelInit.h

int FusedGATProfiling_main(cusp::csr_matrix<int, int, cusp::host_memory> graph,
                           int64_t num_heads, int64_t num_hidden) {
  typedef int32_t Idx;
  typedef float DType;

  MySimpleNDArray<Idx, std::allocator<Idx>> eids_h(
      std::vector<int64_t>{(int64_t)graph.values.size()});
  thrust::sequence<>(eids_h.data.begin(), eids_h.data.end(), 0);

  MyHeteroSeparateCSR<Idx, std::allocator<Idx>> incsr_h(
      std::vector<cusp::csr_matrix<int, int, cusp::host_memory>>{graph},
      eids_h.data);
  MyHeteroSeparateCSR<Idx, std::allocator<Idx>> outcsr_h(incsr_h);

  outcsr_h.Transpose();

  // copy CSR+eid data to device

  MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> incsr(incsr_h);
  MyHeteroSeparateCSR<Idx, thrust::device_allocator<Idx>> outcsr(outcsr_h);

  MySimpleNDArray<DType, thrust::device_allocator<DType>> feat_src =
      GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, num_hidden});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> el =
      GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> er =
      GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> sum =
      GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> exp =
      GenerateRandomNDArray<DType>({incsr.total_num_nnzs, num_heads, 1});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ret =
      GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, num_hidden});

  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_out =
      GenerateRandomNDArray<DType>(
          {incsr.num_rows, num_heads,
           num_hidden});  // TODO: verify if the assumption that the shape is
                          // the same as ret is correct
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_feat_src =
      GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, num_hidden});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_el =
      GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_er =
      GenerateRandomNDArray<DType>({incsr.num_rows, num_heads, 1});

  float slope = 0.2;

  HET::OpPrototyping::FusedGatKernelImpl<Idx, DType>(incsr, feat_src, el, er,
                                                     sum, exp, ret, slope);
  // TODO: check if transposed eid is needed here
  HET::OpPrototyping::BackwardFusedGatKernelImpl<Idx, DType, true>(
      outcsr, feat_src, el, er, sum, exp, ret, grad_out, grad_feat_src, grad_el,
      grad_er, slope);
  HET::OpPrototyping::BackwardFusedGatKernelImpl<Idx, DType, false>(
      outcsr, feat_src, el, er, sum, exp, ret, grad_out, grad_feat_src, grad_el,
      grad_er, slope);
  return 0;
}

int HGTBackPropGradientSMAFusionProfiling_main(
    MyHeteroIntegratedCSR<int32_t, std::allocator<int32_t>> csr_h,
    int64_t num_heads, int64_t num_feat_per_head) {
  typedef int32_t Idx;
  typedef float DType;

  MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> transposed_csr_h(csr_h);

  transposed_csr_h.Transpose();

  // copy CSR+eid data to device
  MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> transposed_csr(
      transposed_csr_h);

  assert(csr_h.num_rels ==
         4);  // memory footprint 50% reduction hack for grad_sm_first_stage
              // only effective for ogbn-mag
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_sm_first_stage =
      GenerateRandomNDArray<DType>(
          {csr_h.num_rows,
           2 /*memory footprint hack only effective for ogbn-mag*/, num_heads,
           num_feat_per_head});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_a =
      GenerateRandomNDArray<DType>({csr_h.total_num_nnzs, num_heads});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_t_neighbour =
      GenerateRandomNDArray<DType>(
          {csr_h.num_rows, num_heads, num_feat_per_head});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> message =
      GenerateRandomNDArray<DType>(
          {csr_h.total_num_nnzs, num_heads, num_feat_per_head});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> sigmas =
      GenerateRandomNDArray<DType>({csr_h.total_num_nnzs, num_heads});

  HET::OpPrototyping::HGTBackPropGradientSMAFusion<Idx, DType>(
      transposed_csr,
      grad_sm_first_stage,  //|V| * N_REL_TYPES * N_HEADS * DIM_PER_HEAD
      grad_a,               // |E| * N_HEADS
      grad_t_neighbour,     //|V| * N_HEADS * DIM_PER_HEAD
      message,              //|E| * N_HEADS * DIM_PER_HEAD
      sigmas);              //|E| * N_HEADS
  return 0;
}

int _HGTExperimental_main(
    MySegmentCSR<int, std::allocator<int>,
                 MyHeteroSeparateCSR<int, std::allocator<int>>> &graph,
    int num_heads, int in_feat, int out_feat) {  // noexcept(false) {
  assert(num_heads == 4);
  typedef int32_t Idx;
  typedef float DType;
  typedef float4 DTypeVec4;

  MySegmentCSR<int, thrust::device_allocator<int>,
               MyHeteroSeparateCSR<int, thrust::device_allocator<int>>>
      deivce_graph = graph;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> node_features =
      GenerateRandomNDArray<DType>({graph.num_rows, num_heads, in_feat});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> weight =
      GenerateRandomNDArray<DType>(
          {graph.num_rels, num_heads, in_feat, out_feat});
  MySimpleNDArray<DTypeVec4, thrust::device_allocator<DTypeVec4>> attention =
      GenerateRandomNDArray<DTypeVec4>({graph.total_num_nnzs, 1});
  HET::OpPrototyping::HGTForwardImpl(deivce_graph, num_heads, in_feat, out_feat,
                                     node_features, weight, attention);
  return 0;
}

// profiling involve both forward and backward in this function
// TODO: put in_feat, out_feat into a hyper parametere structure
int _RGCNLayer1Profiling_main(
    cusp::csr_matrix<int, int, cusp::host_memory> graph, int64_t in_feat,
    int64_t out_feat, bool flagUseMyHyb, bool flagCheckCorrect) {
  typedef int32_t Idx;
  typedef float DType;
  if (flagCheckCorrect) {
    std::cout << "Warning: flagCheckCorrect is true in "
                 "_RGCNLayer1Profiling_main, ignoring flagUseMyHyb and both "
                 "myhyb and the original kernels will be run."
              << std::endl;
  }

  // load data
  MySimpleNDArray<Idx, std::allocator<Idx>> eids_h(
      std::vector<int64_t>{(int64_t)graph.column_indices.size()});
  thrust::sequence<>(eids_h.data.begin(), eids_h.data.end(), 0);
  MySimpleNDArray<Idx, std::allocator<Idx>> transposed_eids_h(eids_h);

  MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> csr_h(
      graph.row_offsets, graph.column_indices, graph.values, eids_h.data);
  MyHeteroIntegratedCSR<Idx, std::allocator<Idx>> transposed_csr_h(csr_h);

  transposed_csr_h.Transpose();

  MyHyb<Idx, std::allocator<Idx>,
        MyHeteroIntegratedCSR<Idx, std::allocator<Idx>>>
      myhyb_h;
  MyHyb<Idx, std::allocator<Idx>,
        MyHeteroIntegratedCSR<Idx, std::allocator<Idx>>>
      transposed_myhyb_h;
  if (flagUseMyHyb || flagCheckCorrect) {
    myhyb_h = IntegratedCSRToHyb_ADHOC_CPU(csr_h, 4, 4, csr_h.num_rows);
    transposed_myhyb_h = IntegratedCSRToHyb_ADHOC_CPU(
        transposed_csr_h, 4, 4, transposed_csr_h.num_rows);
  }
  // copy MyHyb data to device and/or copy CSR+eid data to device
  MyHyb<Idx, thrust::device_allocator<Idx>,
        MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>>>
      myhyb(myhyb_h);
  MyHyb<Idx, thrust::device_allocator<Idx>,
        MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>>>
      transposed_myhyb(transposed_myhyb_h);

  MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr;
  MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> transposed_csr;

  if ((!flagUseMyHyb) || flagCheckCorrect) {
    csr = csr_h;
    transposed_csr = transposed_csr_h;
  }

  MySimpleNDArray<DType, thrust::device_allocator<DType>> hidden =
      GenerateRandomNDArray<DType>(
          {csr_h.num_rows, in_feat});  // TODO: assuming hidden is x. need to
                                       // verify if that is correct
  MySimpleNDArray<DType, thrust::device_allocator<DType>> weight =
      GenerateRandomNDArray<DType>({csr_h.num_rels, in_feat, out_feat});
  // asuming num_bases == num_rels
  MySimpleNDArray<DType, thrust::device_allocator<DType>> norm =
      GenerateRandomNDArray<DType>({csr_h.total_num_nnzs, 1});
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ret;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> ret2;

  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_out =
      GenerateRandomNDArray<DType>(
          {csr_h.num_rows,
           out_feat});  // TODO: verify if the assumption that the shape is the
                        // same as ret is correct
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_hidden2;
  MySimpleNDArray<DType, thrust::device_allocator<DType>> grad_weight2;

  if ((!flagUseMyHyb) || flagCheckCorrect) {
    ret = GenerateRandomNDArray<DType>({csr_h.num_rows, out_feat});
    grad_hidden = GenerateRandomNDArray<DType>({csr_h.total_num_nnzs, in_feat});
    grad_weight =
        GenerateRandomNDArray<DType>({csr_h.num_rels, in_feat, out_feat});
    HET::OpPrototyping::RgcnLayer1Impl<Idx, DType>(csr, hidden, weight, norm,
                                                   ret);
    HET::OpPrototyping::RgcnLayer1BackwardImpl<Idx, DType>(
        transposed_csr, hidden, weight, norm, grad_out, grad_hidden,
        grad_weight);
  }
  if (flagUseMyHyb || flagCheckCorrect) {
    ret2 = GenerateRandomNDArray<DType>({csr_h.num_rows, out_feat});
    grad_hidden2 =
        GenerateRandomNDArray<DType>({csr_h.total_num_nnzs, in_feat});
    grad_weight2 =
        GenerateRandomNDArray<DType>({csr_h.num_rels, in_feat, out_feat});

    RgcnLayer1MyHYBImpl<Idx, DType, 4, 4>(myhyb, hidden, weight, norm, ret2);
    RgcnLayer1BackwardMyHYBImpl<Idx, DType, 4, 4>(transposed_myhyb, hidden,
                                                  weight, norm, grad_out,
                                                  grad_hidden2, grad_weight2);
  }

  if (flagCheckCorrect) {
    std::cout << "check correctness in _RGCNLayer1Profiling_main" << std::endl;
    std::cout << "ret: " << ret.IsEqual<>(ret2) << std::endl;
    std::cout << "grad_hidden: " << grad_hidden.IsEqual<>(grad_hidden2)
              << std::endl;
    std::cout << "grad_weight: " << grad_weight.IsEqual<>(grad_weight2)
              << std::endl;
  }

  return 0;
}

}  // namespace OpPrototyping
}  // namespace HET