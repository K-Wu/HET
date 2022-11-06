#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "MyHyb/MyHyb.h"

std::vector<at::Tensor> convert_integrated_coo_to_separate_coo(
    at::Tensor& integrated_row_idx, at::Tensor& integrated_col_idx,
    at::Tensor& integrated_reltypes, at::Tensor& integrated_eids,
    int64_t num_rows, int64_t num_rels) {
  assert(integrated_row_idx.is_contiguous());
  assert(integrated_col_idx.is_contiguous());
  assert(integrated_reltypes.is_contiguous());
  assert(integrated_eids.is_contiguous());
  assert(integrated_row_idx.device().is_cpu());
  assert(integrated_col_idx.device().is_cpu());
  assert(integrated_reltypes.device().is_cpu());
  assert(integrated_eids.device().is_cpu());

  thrust::host_vector<int64_t> integrated_row_idx_thrust(
      integrated_row_idx.data_ptr<int64_t>(),
      integrated_row_idx.data_ptr<int64_t>() + integrated_row_idx.numel());
  thrust::host_vector<int64_t> integrated_col_idx_thrust(
      integrated_col_idx.data_ptr<int64_t>(),
      integrated_col_idx.data_ptr<int64_t>() + integrated_col_idx.numel());
  thrust::host_vector<int64_t> integrated_reltypes_thrust(
      integrated_reltypes.data_ptr<int64_t>(),
      integrated_reltypes.data_ptr<int64_t>() + integrated_reltypes.numel());
  thrust::host_vector<int64_t> integrated_eids_thrust(
      integrated_eids.data_ptr<int64_t>(),
      integrated_eids.data_ptr<int64_t>() + integrated_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      integrated_row_idx_thrust, integrated_col_idx_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust);

  auto coo_separate = ConvertIntegratedCOOToSeparateCOO<int64_t>(
      integrated_row_idx_thrust, integrated_col_idx_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust, num_rows,
      num_rels);  // {separate_rel_ptr, separate_row_idx,
                  // separate_col_idx, separate_eids}

  at::Tensor separate_relptr =
      at::empty({num_rels + 1}, integrated_reltypes.options());
  at::Tensor separate_row_idx =
      at::empty({integrated_col_idx.numel()}, integrated_col_idx.options());
  at::Tensor separate_col_idx =
      at::empty({integrated_col_idx.numel()}, integrated_col_idx.options());
  at::Tensor separate_eids =
      at::empty({integrated_eids.numel()}, integrated_eids.options());

  assert(separate_relptr.is_contiguous());
  assert(separate_row_idx.is_contiguous());
  assert(separate_col_idx.is_contiguous());
  assert(separate_eids.is_contiguous());

  assert(separate_relptr.device().is_cpu());
  assert(separate_row_idx.device().is_cpu());
  assert(separate_col_idx.device().is_cpu());
  assert(separate_eids.device().is_cpu());

  std::copy(coo_separate[0].begin(), coo_separate[0].end(),
            separate_relptr.data_ptr<int64_t>());
  std::copy(coo_separate[1].begin(), coo_separate[1].end(),
            separate_row_idx.data_ptr<int64_t>());
  std::copy(coo_separate[2].begin(), coo_separate[2].end(),
            separate_col_idx.data_ptr<int64_t>());
  std::copy(coo_separate[3].begin(), coo_separate[3].end(),
            separate_eids.data_ptr<int64_t>());

  std::vector<at::Tensor> result = {separate_relptr, separate_row_idx,
                                    separate_col_idx, separate_eids};
  return result;
}

std::vector<at::Tensor> convert_integrated_coo_to_separate_csr(
    at::Tensor& integrated_row_idx, at::Tensor& integrated_col_idx,
    at::Tensor& integrated_reltypes, at::Tensor& integrated_eids,
    int64_t num_rows, int64_t num_rels) {
  assert(integrated_row_idx.is_contiguous());
  assert(integrated_col_idx.is_contiguous());
  assert(integrated_reltypes.is_contiguous());
  assert(integrated_eids.is_contiguous());
  assert(integrated_row_idx.device().is_cpu());
  assert(integrated_col_idx.device().is_cpu());
  assert(integrated_reltypes.device().is_cpu());
  assert(integrated_eids.device().is_cpu());

  thrust::host_vector<int64_t> integrated_row_idx_thrust(
      integrated_row_idx.data_ptr<int64_t>(),
      integrated_row_idx.data_ptr<int64_t>() + integrated_row_idx.numel());
  thrust::host_vector<int64_t> integrated_col_idx_thrust(
      integrated_col_idx.data_ptr<int64_t>(),
      integrated_col_idx.data_ptr<int64_t>() + integrated_col_idx.numel());
  thrust::host_vector<int64_t> integrated_reltypes_thrust(
      integrated_reltypes.data_ptr<int64_t>(),
      integrated_reltypes.data_ptr<int64_t>() + integrated_reltypes.numel());
  thrust::host_vector<int64_t> integrated_eids_thrust(
      integrated_eids.data_ptr<int64_t>(),
      integrated_eids.data_ptr<int64_t>() + integrated_eids.numel());

  auto csr_separate = ConvertIntegratedCOOToSeparateCSR<int64_t>(
      integrated_row_idx_thrust, integrated_col_idx_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust, num_rows, num_rels);

  at::Tensor separate_relptr =
      at::empty({num_rels + 1}, integrated_reltypes.options());
  at::Tensor separate_rowptr =
      at::empty({num_rows * num_rels + 1}, integrated_row_idx.options());
  at::Tensor separate_col_idx =
      at::empty({integrated_col_idx.numel()}, integrated_col_idx.options());
  at::Tensor separate_eids =
      at::empty({integrated_eids.numel()}, integrated_eids.options());

  assert(separate_relptr.is_contiguous());
  assert(separate_rowptr.is_contiguous());
  assert(separate_col_idx.is_contiguous());
  assert(separate_eids.is_contiguous());

  assert(separate_relptr.device().is_cpu());
  assert(separate_rowptr.device().is_cpu());
  assert(separate_col_idx.device().is_cpu());
  assert(separate_eids.device().is_cpu());

  thrust::copy(csr_separate[0].begin(), csr_separate[0].end(),
               separate_relptr.data_ptr<int64_t>());
  thrust::copy(csr_separate[1].begin(), csr_separate[1].end(),
               separate_rowptr.data_ptr<int64_t>());
  thrust::copy(csr_separate[2].begin(), csr_separate[2].end(),

               separate_col_idx.data_ptr<int64_t>());
  thrust::copy(csr_separate[3].begin(), csr_separate[3].end(),
               separate_eids.data_ptr<int64_t>());
  std::vector<at::Tensor> result = {separate_relptr, separate_rowptr,
                                    separate_col_idx, separate_eids};
  return result;
}

std::vector<at::Tensor> convert_integrated_csr_to_separate_csr(
    at::Tensor& integrated_rowptr, at::Tensor& integrated_col_idx,
    at::Tensor& integrated_reltypes, at::Tensor& integrated_eids) {
  assert(integrated_rowptr.is_contiguous());
  assert(integrated_col_idx.is_contiguous());
  assert(integrated_reltypes.is_contiguous());
  assert(integrated_eids.is_contiguous());
  assert(integrated_rowptr.device().is_cpu());
  assert(integrated_col_idx.device().is_cpu());
  assert(integrated_reltypes.device().is_cpu());
  assert(integrated_eids.device().is_cpu());

  thrust::host_vector<int64_t> integrated_rowptr_thrust(
      integrated_rowptr.data_ptr<int64_t>(),
      integrated_rowptr.data_ptr<int64_t>() + integrated_rowptr.numel());
  thrust::host_vector<int64_t> integrated_col_idx_thrust(
      integrated_col_idx.data_ptr<int64_t>(),
      integrated_col_idx.data_ptr<int64_t>() + integrated_col_idx.numel());
  thrust::host_vector<int64_t> integrated_reltypes_thrust(
      integrated_reltypes.data_ptr<int64_t>(),
      integrated_reltypes.data_ptr<int64_t>() + integrated_reltypes.numel());
  thrust::host_vector<int64_t> integrated_eids_thrust(
      integrated_eids.data_ptr<int64_t>(),
      integrated_eids.data_ptr<int64_t>() + integrated_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      integrated_rowptr_thrust, integrated_col_idx_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust);

  auto csr_separate =
      ToSeparateCSR<std::allocator<int64_t>, std::allocator<int64_t>, int64_t>(
          csr);

  at::Tensor separate_relptr =
      at::empty({csr_separate.num_rels + 1}, integrated_reltypes.options());
  at::Tensor separate_rowptr =
      at::empty({csr_separate.num_rows * csr_separate.num_rels + 1},
                integrated_rowptr.options());
  at::Tensor separate_col_idx =
      at::empty({integrated_col_idx.numel()}, integrated_col_idx.options());
  at::Tensor separate_eids =
      at::empty({integrated_eids.numel()}, integrated_eids.options());

  assert(separate_relptr.is_contiguous());
  assert(separate_rowptr.is_contiguous());
  assert(separate_col_idx.is_contiguous());
  assert(separate_eids.is_contiguous());

  assert(separate_relptr.device().is_cpu());
  assert(separate_rowptr.device().is_cpu());
  assert(separate_col_idx.device().is_cpu());
  assert(separate_eids.device().is_cpu());

  separate_relptr[0] = 0;
  for (size_t rel_idx = 0; rel_idx < csr_separate.num_nnzs.size(); rel_idx++) {
    separate_relptr[rel_idx + 1] =
        separate_relptr[rel_idx] + csr_separate.num_nnzs[rel_idx];
  }

  thrust::copy(csr_separate.row_ptr.begin(), csr_separate.row_ptr.end(),
               separate_rowptr.data_ptr<int64_t>());
  thrust::copy(csr_separate.col_idx.begin(), csr_separate.col_idx.end(),

               separate_col_idx.data_ptr<int64_t>());
  thrust::copy(csr_separate.eids.begin(), csr_separate.eids.end(),
               separate_eids.data_ptr<int64_t>());
  std::vector<at::Tensor> result = {separate_relptr, separate_rowptr,
                                    separate_col_idx, separate_eids};
  return result;
}

std::vector<at::Tensor> convert_integrated_csr_to_separate_coo(
    at::Tensor& integrated_rowptr, at::Tensor& integrated_col_idx,
    at::Tensor& integrated_reltypes, at::Tensor& integrated_eids) {
  assert(integrated_rowptr.is_contiguous());
  assert(integrated_col_idx.is_contiguous());
  assert(integrated_reltypes.is_contiguous());
  assert(integrated_eids.is_contiguous());
  assert(integrated_rowptr.device().is_cpu());
  assert(integrated_col_idx.device().is_cpu());
  assert(integrated_reltypes.device().is_cpu());
  assert(integrated_eids.device().is_cpu());

  thrust::host_vector<int64_t> integrated_rowptr_thrust(
      integrated_rowptr.data_ptr<int64_t>(),
      integrated_rowptr.data_ptr<int64_t>() + integrated_rowptr.numel());
  thrust::host_vector<int64_t> integrated_col_idx_thrust(
      integrated_col_idx.data_ptr<int64_t>(),
      integrated_col_idx.data_ptr<int64_t>() + integrated_col_idx.numel());
  thrust::host_vector<int64_t> integrated_reltypes_thrust(
      integrated_reltypes.data_ptr<int64_t>(),
      integrated_reltypes.data_ptr<int64_t>() + integrated_reltypes.numel());
  thrust::host_vector<int64_t> integrated_eids_thrust(
      integrated_eids.data_ptr<int64_t>(),
      integrated_eids.data_ptr<int64_t>() + integrated_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      integrated_rowptr_thrust, integrated_col_idx_thrust,
      integrated_reltypes_thrust, integrated_eids_thrust);

  auto coo_separate =
      ToSeparateCOO<int64_t>(csr);  // {separate_rel_ptr, separate_row_idx,
                                    // separate_col_idx, separate_eids}

  at::Tensor separate_relptr =
      at::empty({csr.num_rels + 1}, integrated_reltypes.options());
  at::Tensor separate_row_idx =
      at::empty({integrated_col_idx.numel()}, integrated_col_idx.options());
  at::Tensor separate_col_idx =
      at::empty({integrated_col_idx.numel()}, integrated_col_idx.options());
  at::Tensor separate_eids =
      at::empty({integrated_eids.numel()}, integrated_eids.options());

  assert(separate_relptr.is_contiguous());
  assert(separate_row_idx.is_contiguous());
  assert(separate_col_idx.is_contiguous());
  assert(separate_eids.is_contiguous());

  assert(separate_relptr.device().is_cpu());
  assert(separate_row_idx.device().is_cpu());
  assert(separate_col_idx.device().is_cpu());
  assert(separate_eids.device().is_cpu());

  std::copy(coo_separate[0].begin(), coo_separate[0].end(),
            separate_relptr.data_ptr<int64_t>());
  std::copy(coo_separate[1].begin(), coo_separate[1].end(),
            separate_row_idx.data_ptr<int64_t>());
  std::copy(coo_separate[2].begin(), coo_separate[2].end(),
            separate_col_idx.data_ptr<int64_t>());
  std::copy(coo_separate[3].begin(), coo_separate[3].end(),
            separate_eids.data_ptr<int64_t>());

  std::vector<at::Tensor> result = {separate_relptr, separate_row_idx,
                                    separate_col_idx, separate_eids};
  return result;
}

std::vector<at::Tensor> transpose_csr(at::Tensor& csr_rowptr,
                                      at::Tensor& csr_col_idx,
                                      at::Tensor& csr_reltypes,
                                      at::Tensor& csr_eids) {
  // NB: graphiler, seastar by default uses int64_t
  assert(csr_rowptr.is_contiguous());
  assert(csr_col_idx.is_contiguous());
  assert(csr_reltypes.is_contiguous());
  assert(csr_eids.is_contiguous());
  assert(csr_rowptr.device().is_cpu());
  assert(csr_col_idx.device().is_cpu());
  assert(csr_reltypes.device().is_cpu());
  assert(csr_eids.device().is_cpu());

  thrust::host_vector<int64_t> csr_rowptr_thrust(
      csr_rowptr.data_ptr<int64_t>(),
      csr_rowptr.data_ptr<int64_t>() + csr_rowptr.numel());
  thrust::host_vector<int64_t> csr_col_idx_thrust(
      csr_col_idx.data_ptr<int64_t>(),
      csr_col_idx.data_ptr<int64_t>() + csr_col_idx.numel());
  thrust::host_vector<int64_t> csr_reltypes_thrust(
      csr_reltypes.data_ptr<int64_t>(),
      csr_reltypes.data_ptr<int64_t>() + csr_reltypes.numel());
  thrust::host_vector<int64_t> csr_eids_thrust(
      csr_eids.data_ptr<int64_t>(),
      csr_eids.data_ptr<int64_t>() + csr_eids.numel());

  MyHeteroIntegratedCSR<int64_t, std::allocator<int64_t>> csr(
      csr_rowptr_thrust, csr_col_idx_thrust, csr_reltypes_thrust,
      csr_eids_thrust);
  csr.Transpose();

  at::Tensor transposed_rowptr =
      at::empty({csr_rowptr.numel()}, csr_rowptr.options());
  at::Tensor transposed_col_idx =
      at::empty({csr_col_idx.numel()}, csr_col_idx.options());
  at::Tensor transposed_reltypes =
      at::empty({csr_reltypes.numel()}, csr_reltypes.options());
  at::Tensor transposed_eids =
      at::empty({csr_eids.numel()}, csr_eids.options());
  assert(transposed_rowptr.is_contiguous());
  assert(transposed_col_idx.is_contiguous());
  assert(transposed_reltypes.is_contiguous());
  assert(transposed_eids.is_contiguous());
  assert(transposed_rowptr.device().is_cpu());
  assert(transposed_col_idx.device().is_cpu());
  assert(transposed_reltypes.device().is_cpu());
  assert(transposed_eids.device().is_cpu());

  std::copy(csr.row_ptr.begin(), csr.row_ptr.end(),
            transposed_rowptr.data_ptr<int64_t>());
  std::copy(csr.col_idx.begin(), csr.col_idx.end(),
            transposed_col_idx.data_ptr<int64_t>());
  std::copy(csr.rel_type.begin(), csr.rel_type.end(),
            transposed_reltypes.data_ptr<int64_t>());
  std::copy(csr.eids.begin(), csr.eids.end(),
            transposed_eids.data_ptr<int64_t>());

  std::vector<at::Tensor> result = {transposed_rowptr, transposed_col_idx,
                                    transposed_reltypes, transposed_eids};
  return result;
}