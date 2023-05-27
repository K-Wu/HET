#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "MyHyb/MyHyb.h"

// TODO: KWU: allow more dtype than int64_t
std::vector<at::Tensor> convert_integrated_coo_to_separate_coo(
    at::Tensor& integrated_row_indices, at::Tensor& integrated_col_indices,
    at::Tensor& integrated_reltypes, at::Tensor& integrated_eids,
    int64_t num_rows, int64_t num_rels) {
  assert(integrated_row_indices.is_contiguous());
  assert(integrated_col_indices.is_contiguous());
  assert(integrated_reltypes.is_contiguous());
  assert(integrated_eids.is_contiguous());
  assert(integrated_row_indices.device().is_cpu());
  assert(integrated_col_indices.device().is_cpu());
  assert(integrated_reltypes.device().is_cpu());
  assert(integrated_eids.device().is_cpu());

  thrust::host_vector<int64_t> integrated_row_indices_thrust(
      integrated_row_indices.data_ptr<int64_t>(),
      integrated_row_indices.data_ptr<int64_t>() +
          integrated_row_indices.numel());
  thrust::host_vector<int64_t> integrated_col_indices_thrust(
      integrated_col_indices.data_ptr<int64_t>(),
      integrated_col_indices.data_ptr<int64_t>() +
          integrated_col_indices.numel());
  thrust::host_vector<int64_t> integrated_reltypes_thrust(
      integrated_reltypes.data_ptr<int64_t>(),
      integrated_reltypes.data_ptr<int64_t>() + integrated_reltypes.numel());
  thrust::host_vector<int64_t> integrated_eids_thrust(
      integrated_eids.data_ptr<int64_t>(),
      integrated_eids.data_ptr<int64_t>() + integrated_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      integrated_row_indices_thrust, integrated_col_indices_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust);

  auto coo_separate = ConvertIntegratedCOOToSeparateCOO<int64_t>(
      integrated_row_indices_thrust, integrated_col_indices_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust, num_rows, num_rels);

  at::Tensor separate_rel_ptrs =
      at::empty({num_rels + 1}, integrated_reltypes.options());
  at::Tensor separate_row_indices = at::empty({integrated_col_indices.numel()},
                                              integrated_col_indices.options());
  at::Tensor separate_col_indices = at::empty({integrated_col_indices.numel()},
                                              integrated_col_indices.options());
  at::Tensor separate_eids =
      at::empty({integrated_eids.numel()}, integrated_eids.options());

  assert(separate_rel_ptrs.is_contiguous());
  assert(separate_row_indices.is_contiguous());
  assert(separate_col_indices.is_contiguous());
  assert(separate_eids.is_contiguous());

  assert(separate_rel_ptrs.device().is_cpu());
  assert(separate_row_indices.device().is_cpu());
  assert(separate_col_indices.device().is_cpu());
  assert(separate_eids.device().is_cpu());

  std::copy(coo_separate[0].begin(), coo_separate[0].end(),
            separate_rel_ptrs.data_ptr<int64_t>());
  std::copy(coo_separate[1].begin(), coo_separate[1].end(),
            separate_row_indices.data_ptr<int64_t>());
  std::copy(coo_separate[2].begin(), coo_separate[2].end(),
            separate_col_indices.data_ptr<int64_t>());
  std::copy(coo_separate[3].begin(), coo_separate[3].end(),
            separate_eids.data_ptr<int64_t>());

  std::vector<at::Tensor> result = {separate_rel_ptrs, separate_row_indices,
                                    separate_col_indices, separate_eids};
  return result;
}

std::vector<at::Tensor> convert_integrated_coo_to_separate_csr(
    at::Tensor& integrated_row_indices, at::Tensor& integrated_col_indices,
    at::Tensor& integrated_reltypes, at::Tensor& integrated_eids,
    int64_t num_rows, int64_t num_rels) {
  assert(integrated_row_indices.is_contiguous());
  assert(integrated_col_indices.is_contiguous());
  assert(integrated_reltypes.is_contiguous());
  assert(integrated_eids.is_contiguous());
  assert(integrated_row_indices.device().is_cpu());
  assert(integrated_col_indices.device().is_cpu());
  assert(integrated_reltypes.device().is_cpu());
  assert(integrated_eids.device().is_cpu());

  thrust::host_vector<int64_t> integrated_row_indices_thrust(
      integrated_row_indices.data_ptr<int64_t>(),
      integrated_row_indices.data_ptr<int64_t>() +
          integrated_row_indices.numel());
  thrust::host_vector<int64_t> integrated_col_indices_thrust(
      integrated_col_indices.data_ptr<int64_t>(),
      integrated_col_indices.data_ptr<int64_t>() +
          integrated_col_indices.numel());
  thrust::host_vector<int64_t> integrated_reltypes_thrust(
      integrated_reltypes.data_ptr<int64_t>(),
      integrated_reltypes.data_ptr<int64_t>() + integrated_reltypes.numel());
  thrust::host_vector<int64_t> integrated_eids_thrust(
      integrated_eids.data_ptr<int64_t>(),
      integrated_eids.data_ptr<int64_t>() + integrated_eids.numel());

  auto csr_separate = ConvertIntegratedCOOToSeparateCSR<int64_t>(
      integrated_row_indices_thrust, integrated_col_indices_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust, num_rows, num_rels);

  at::Tensor separate_rel_ptrs =
      at::empty({num_rels + 1}, integrated_reltypes.options());
  at::Tensor separate_row_ptrs =
      at::empty({num_rows * num_rels + 1}, integrated_row_indices.options());
  at::Tensor separate_col_indices = at::empty({integrated_col_indices.numel()},
                                              integrated_col_indices.options());
  at::Tensor separate_eids =
      at::empty({integrated_eids.numel()}, integrated_eids.options());

  assert(separate_rel_ptrs.is_contiguous());
  assert(separate_row_ptrs.is_contiguous());
  assert(separate_col_indices.is_contiguous());
  assert(separate_eids.is_contiguous());

  assert(separate_rel_ptrs.device().is_cpu());
  assert(separate_row_ptrs.device().is_cpu());
  assert(separate_col_indices.device().is_cpu());
  assert(separate_eids.device().is_cpu());

  thrust::copy(csr_separate[0].begin(), csr_separate[0].end(),
               separate_rel_ptrs.data_ptr<int64_t>());
  thrust::copy(csr_separate[1].begin(), csr_separate[1].end(),
               separate_row_ptrs.data_ptr<int64_t>());
  thrust::copy(csr_separate[2].begin(), csr_separate[2].end(),

               separate_col_indices.data_ptr<int64_t>());
  thrust::copy(csr_separate[3].begin(), csr_separate[3].end(),
               separate_eids.data_ptr<int64_t>());
  std::vector<at::Tensor> result = {separate_rel_ptrs, separate_row_ptrs,
                                    separate_col_indices, separate_eids};
  return result;
}

std::vector<at::Tensor> convert_integrated_csr_to_separate_csr(
    at::Tensor& integrated_row_ptrs, at::Tensor& integrated_col_indices,
    at::Tensor& integrated_reltypes, at::Tensor& integrated_eids) {
  assert(integrated_row_ptrs.is_contiguous());
  assert(integrated_col_indices.is_contiguous());
  assert(integrated_reltypes.is_contiguous());
  assert(integrated_eids.is_contiguous());
  assert(integrated_row_ptrs.device().is_cpu());
  assert(integrated_col_indices.device().is_cpu());
  assert(integrated_reltypes.device().is_cpu());
  assert(integrated_eids.device().is_cpu());

  thrust::host_vector<int64_t> integrated_row_ptrs_thrust(
      integrated_row_ptrs.data_ptr<int64_t>(),
      integrated_row_ptrs.data_ptr<int64_t>() + integrated_row_ptrs.numel());
  thrust::host_vector<int64_t> integrated_col_indices_thrust(
      integrated_col_indices.data_ptr<int64_t>(),
      integrated_col_indices.data_ptr<int64_t>() +
          integrated_col_indices.numel());
  thrust::host_vector<int64_t> integrated_reltypes_thrust(
      integrated_reltypes.data_ptr<int64_t>(),
      integrated_reltypes.data_ptr<int64_t>() + integrated_reltypes.numel());
  thrust::host_vector<int64_t> integrated_eids_thrust(
      integrated_eids.data_ptr<int64_t>(),
      integrated_eids.data_ptr<int64_t>() + integrated_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      integrated_row_ptrs_thrust, integrated_col_indices_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust);

  auto csr_separate =
      ToSeparateCSR<std::allocator<int64_t>, std::allocator<int64_t>, int64_t>(
          csr);

  at::Tensor separate_rel_ptrs =
      at::empty({csr_separate.num_rels + 1}, integrated_reltypes.options());
  at::Tensor separate_row_ptrs =
      at::empty({csr_separate.num_rows * csr_separate.num_rels + 1},
                integrated_row_ptrs.options());
  at::Tensor separate_col_indices = at::empty({integrated_col_indices.numel()},
                                              integrated_col_indices.options());
  at::Tensor separate_eids =
      at::empty({integrated_eids.numel()}, integrated_eids.options());

  assert(separate_rel_ptrs.is_contiguous());
  assert(separate_row_ptrs.is_contiguous());
  assert(separate_col_indices.is_contiguous());
  assert(separate_eids.is_contiguous());

  assert(separate_rel_ptrs.device().is_cpu());
  assert(separate_row_ptrs.device().is_cpu());
  assert(separate_col_indices.device().is_cpu());
  assert(separate_eids.device().is_cpu());

  separate_rel_ptrs[0] = 0;
  for (size_t rel_idx = 0; rel_idx < csr_separate.num_nnzs.size(); rel_idx++) {
    separate_rel_ptrs[rel_idx + 1] =
        separate_rel_ptrs[rel_idx] + csr_separate.num_nnzs[rel_idx];
  }

  thrust::copy(csr_separate.row_ptrs.begin(), csr_separate.row_ptrs.end(),
               separate_row_ptrs.data_ptr<int64_t>());
  thrust::copy(csr_separate.col_indices.begin(), csr_separate.col_indices.end(),

               separate_col_indices.data_ptr<int64_t>());
  thrust::copy(csr_separate.eids.begin(), csr_separate.eids.end(),
               separate_eids.data_ptr<int64_t>());
  std::vector<at::Tensor> result = {separate_rel_ptrs, separate_row_ptrs,
                                    separate_col_indices, separate_eids};
  return result;
}

std::vector<at::Tensor> convert_integrated_csr_to_separate_coo(
    at::Tensor& integrated_row_ptrs, at::Tensor& integrated_col_indices,
    at::Tensor& integrated_reltypes, at::Tensor& integrated_eids) {
  assert(integrated_row_ptrs.is_contiguous());
  assert(integrated_col_indices.is_contiguous());
  assert(integrated_reltypes.is_contiguous());
  assert(integrated_eids.is_contiguous());
  assert(integrated_row_ptrs.device().is_cpu());
  assert(integrated_col_indices.device().is_cpu());
  assert(integrated_reltypes.device().is_cpu());
  assert(integrated_eids.device().is_cpu());

  thrust::host_vector<int64_t> integrated_row_ptrs_thrust(
      integrated_row_ptrs.data_ptr<int64_t>(),
      integrated_row_ptrs.data_ptr<int64_t>() + integrated_row_ptrs.numel());
  thrust::host_vector<int64_t> integrated_col_indices_thrust(
      integrated_col_indices.data_ptr<int64_t>(),
      integrated_col_indices.data_ptr<int64_t>() +
          integrated_col_indices.numel());
  thrust::host_vector<int64_t> integrated_reltypes_thrust(
      integrated_reltypes.data_ptr<int64_t>(),
      integrated_reltypes.data_ptr<int64_t>() + integrated_reltypes.numel());
  thrust::host_vector<int64_t> integrated_eids_thrust(
      integrated_eids.data_ptr<int64_t>(),
      integrated_eids.data_ptr<int64_t>() + integrated_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      integrated_row_ptrs_thrust, integrated_col_indices_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust);

  auto coo_separate =
      ToSeparateCOO<int64_t>(csr);  // {separate_rel_ptr, separate_row_indices,
                                    // separate_col_indices, separate_eids}

  at::Tensor separate_rel_ptrs =
      at::empty({csr.num_rels + 1}, integrated_reltypes.options());
  at::Tensor separate_row_indices = at::empty({integrated_col_indices.numel()},
                                              integrated_col_indices.options());
  at::Tensor separate_col_indices = at::empty({integrated_col_indices.numel()},
                                              integrated_col_indices.options());
  at::Tensor separate_eids =
      at::empty({integrated_eids.numel()}, integrated_eids.options());

  assert(separate_rel_ptrs.is_contiguous());
  assert(separate_row_indices.is_contiguous());
  assert(separate_col_indices.is_contiguous());
  assert(separate_eids.is_contiguous());

  assert(separate_rel_ptrs.device().is_cpu());
  assert(separate_row_indices.device().is_cpu());
  assert(separate_col_indices.device().is_cpu());
  assert(separate_eids.device().is_cpu());

  std::copy(coo_separate[0].begin(), coo_separate[0].end(),
            separate_rel_ptrs.data_ptr<int64_t>());
  std::copy(coo_separate[1].begin(), coo_separate[1].end(),
            separate_row_indices.data_ptr<int64_t>());
  std::copy(coo_separate[2].begin(), coo_separate[2].end(),
            separate_col_indices.data_ptr<int64_t>());
  std::copy(coo_separate[3].begin(), coo_separate[3].end(),
            separate_eids.data_ptr<int64_t>());

  std::vector<at::Tensor> result = {separate_rel_ptrs, separate_row_indices,
                                    separate_col_indices, separate_eids};
  return result;
}

std::vector<at::Tensor> transpose_csr(at::Tensor& csr_row_ptrs,
                                      at::Tensor& csr_col_indices,
                                      at::Tensor& csr_reltypes,
                                      at::Tensor& csr_eids) {
  // NB: graphiler, seastar by default uses int64_t
  assert(csr_row_ptrs.is_contiguous());
  assert(csr_col_indices.is_contiguous());
  assert(csr_reltypes.is_contiguous());
  assert(csr_eids.is_contiguous());
  assert(csr_row_ptrs.device().is_cpu());
  assert(csr_col_indices.device().is_cpu());
  assert(csr_reltypes.device().is_cpu());
  assert(csr_eids.device().is_cpu());

  thrust::host_vector<int64_t> csr_row_ptrs_thrust(
      csr_row_ptrs.data_ptr<int64_t>(),
      csr_row_ptrs.data_ptr<int64_t>() + csr_row_ptrs.numel());
  thrust::host_vector<int64_t> csr_col_indices_thrust(
      csr_col_indices.data_ptr<int64_t>(),
      csr_col_indices.data_ptr<int64_t>() + csr_col_indices.numel());
  thrust::host_vector<int64_t> csr_reltypes_thrust(
      csr_reltypes.data_ptr<int64_t>(),
      csr_reltypes.data_ptr<int64_t>() + csr_reltypes.numel());
  thrust::host_vector<int64_t> csr_eids_thrust(
      csr_eids.data_ptr<int64_t>(),
      csr_eids.data_ptr<int64_t>() + csr_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      csr_row_ptrs_thrust, csr_col_indices_thrust, csr_reltypes_thrust,
      csr_eids_thrust);
  csr.Transpose();

  at::Tensor transposed_row_ptrs =
      at::empty({csr_row_ptrs.numel()}, csr_row_ptrs.options());
  at::Tensor transposed_col_indices =
      at::empty({csr_col_indices.numel()}, csr_col_indices.options());
  at::Tensor transposed_reltypes =
      at::empty({csr_reltypes.numel()}, csr_reltypes.options());
  at::Tensor transposed_eids =
      at::empty({csr_eids.numel()}, csr_eids.options());
  assert(transposed_row_ptrs.is_contiguous());
  assert(transposed_col_indices.is_contiguous());
  assert(transposed_reltypes.is_contiguous());
  assert(transposed_eids.is_contiguous());
  assert(transposed_row_ptrs.device().is_cpu());
  assert(transposed_col_indices.device().is_cpu());
  assert(transposed_reltypes.device().is_cpu());
  assert(transposed_eids.device().is_cpu());

  std::copy(csr.row_ptrs.begin(), csr.row_ptrs.end(),
            transposed_row_ptrs.data_ptr<int64_t>());
  std::copy(csr.col_indices.begin(), csr.col_indices.end(),
            transposed_col_indices.data_ptr<int64_t>());
  std::copy(csr.rel_type.begin(), csr.rel_type.end(),
            transposed_reltypes.data_ptr<int64_t>());
  std::copy(csr.eids.begin(), csr.eids.end(),
            transposed_eids.data_ptr<int64_t>());

  std::vector<at::Tensor> result = {transposed_row_ptrs, transposed_col_indices,
                                    transposed_reltypes, transposed_eids};
  return result;
}

TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // Data Converters
  m.def("transpose_csr", transpose_csr);
  m.def("convert_integrated_csr_to_separate_csr",
        convert_integrated_csr_to_separate_csr);
  m.def("convert_integrated_csr_to_separate_coo",
        convert_integrated_csr_to_separate_coo);
  m.def("convert_integrated_coo_to_separate_csr",
        convert_integrated_coo_to_separate_csr);
  m.def("convert_integrated_coo_to_separate_coo",
        convert_integrated_coo_to_separate_coo);
}
