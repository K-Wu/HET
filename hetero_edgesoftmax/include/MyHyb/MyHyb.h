#pragma once
#include <cusp/csr_matrix.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <map>
#include <numeric>
#include <optional>
#include <set>
// #include "../hetero_edgesoftmax.h"
// #include <optional>

#ifdef MyHyb_NONEXISTENT_ELEMENT
#error "MyHyb_DEFINE_CONSTANT is already defined"
#else
#define MyHyb_NONEXISTENT_ELEMENT -1
#endif

template <class InputIt, class T>
T my_accumulate(InputIt first, InputIt last, T init) {
  for (; first != last; ++first) {
    init += *first;
  }
  return init;
}

template <typename IdxType>
thrust::host_vector<IdxType> TransposeCSR(
    thrust::detail::vector_base<IdxType, std::allocator<IdxType>> &row_ptrs,
    thrust::detail::vector_base<IdxType, std::allocator<IdxType>>
        &col_indices) {
  // transpose the csr matrix, and return the permutation array so that rel_type
  // in integratedCSR and eid in FusedGAT can be mapped to the new order using
  // the permutation array.
  thrust::host_vector<IdxType> permutation(col_indices.size());
  thrust::sequence<>(permutation.begin(), permutation.end(), 0);
  thrust::host_vector<IdxType> new_row_ptrs(row_ptrs.size());
  thrust::host_vector<IdxType> new_col_indices(col_indices.size());

  std::map<IdxType, std::vector<std::pair<IdxType, IdxType>>> col_row_map;

  for (int64_t i = 0; i < row_ptrs.size() - 1;
       i++) {  // TODO: possible loss of precision here as loop variable is
               // int64_t but the indices in matrix is IdxType
    for (int64_t j = row_ptrs[i]; j < row_ptrs[i + 1]; j++) {
      assert(col_indices[j] < row_ptrs.size() - 1);
      col_row_map[col_indices[j]].push_back(std::make_pair(i, permutation[j]));
    }
  }

  new_row_ptrs[0] = 0;
  for (int64_t idxNode = 0; idxNode < row_ptrs.size() - 1;
       idxNode++) {  // assert num_rows == num_cols
    new_row_ptrs[idxNode + 1] =
        new_row_ptrs[idxNode] + col_row_map[idxNode].size();
    for (int64_t idxEdgeForCurrNode = 0;
         idxEdgeForCurrNode < col_row_map[idxNode].size();
         idxEdgeForCurrNode++) {
      new_col_indices[new_row_ptrs[idxNode] + idxEdgeForCurrNode] =
          col_row_map[idxNode][idxEdgeForCurrNode].first;
      permutation[new_row_ptrs[idxNode] + idxEdgeForCurrNode] =
          col_row_map[idxNode][idxEdgeForCurrNode].second;
    }
  }

  thrust::copy(new_row_ptrs.begin(), new_row_ptrs.end(), row_ptrs.begin());
  thrust::copy(new_col_indices.begin(), new_col_indices.end(),
               col_indices.begin());
  return permutation;
}

template <typename IdxType, typename Alloc>
bool IsDeviceVector(const thrust::detail::vector_base<IdxType, Alloc> &vec) {
  return std::is_same<typename thrust::device_vector<IdxType>::allocator_type,
                      Alloc>::value;
}

template <typename IdxType, typename Alloc>
bool IsHostVector(const thrust::detail::vector_base<IdxType, Alloc> &vec) {
  return std::is_same<typename thrust::host_vector<IdxType>::allocator_type,
                      Alloc>::value;
}

// TODO: implement unique src/dst index API for HGT model
template <typename IdxType, typename Alloc>
class MyHeteroIntegratedCSR;

// TODO: implement unique src/dst index API for HGT model
template <typename IdxType, typename Alloc>
class MyHeteroSeparateCSR {
 public:
  int64_t num_rows;
  int64_t num_cols;
  int64_t num_rels;
  std::vector<int64_t> num_nnzs;
  int total_num_nnzs;
  thrust::detail::vector_base<IdxType, Alloc> row_ptrs;
  thrust::detail::vector_base<IdxType, Alloc> col_indices;
  thrust::detail::vector_base<IdxType, Alloc> eids;

  MyHeteroSeparateCSR() = default;
  // std::allocator<T>
  // thrust::device_allocator<T>
  template <typename OtherAlloc>
  MyHeteroSeparateCSR(
      const int64_t num_rows, const int64_t num_cols, const int64_t num_rels,
      const std::vector<int64_t> &num_nnzs,
      const thrust::detail::vector_base<IdxType, OtherAlloc> &row_ptrs,
      const thrust::detail::vector_base<IdxType, OtherAlloc> &col_indices,
      const thrust::detail::vector_base<IdxType, OtherAlloc> &eids) {
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->num_rels = num_rels;
    this->num_nnzs = num_nnzs;
    this->total_num_nnzs =
        my_accumulate<>(num_nnzs.begin(), num_nnzs.end(), 0LL);
    this->row_ptrs = row_ptrs;
    this->col_indices = col_indices;
    this->eids = eids;
  }

  template <typename ValueType, typename MemorySpace, typename OtherAlloc>
  MyHeteroSeparateCSR(
      const std::vector<cusp::csr_matrix<IdxType, ValueType, MemorySpace>>
          &cusp_csrs,
      const thrust::detail::vector_base<IdxType, OtherAlloc> &eids) {
    this->eids = eids;
    this->num_nnzs = std::vector<int64_t>(cusp_csrs.size(), 0);
    this->num_rows = 0;
    this->num_cols = 0;
    for (int64_t csr_idx = 0; csr_idx < cusp_csrs.size(); csr_idx++) {
      this->num_rows =
          std::max(this->num_rows, (int64_t)cusp_csrs[csr_idx].num_rows);
      this->num_cols =
          std::max(this->num_cols, (int64_t)cusp_csrs[csr_idx].num_cols);
      this->num_nnzs[csr_idx] = cusp_csrs[csr_idx].num_entries;
    }
    this->num_rels = cusp_csrs.size();
    this->total_num_nnzs = my_accumulate<>(num_nnzs.begin(), num_nnzs.end(), 0);
    this->row_ptrs = thrust::detail::vector_base<IdxType, Alloc>(
        this->num_rels * this->num_rows + 1, 0LL);
    this->col_indices =
        thrust::detail::vector_base<IdxType, Alloc>(this->total_num_nnzs, 0);

    for (int64_t IdxRelationship = 0; IdxRelationship < this->num_rels;
         IdxRelationship++) {
      for (int64_t IdxRow = 0; IdxRow < this->num_rows; IdxRow++) {
        if (cusp_csrs[IdxRelationship].row_offsets.size() <= IdxRow) {
          // this csr has less rows than the current row index
          this->row_ptrs[IdxRow + IdxRelationship * this->num_rows + 1] =
              this->row_ptrs[IdxRow + IdxRelationship * this->num_rows];
        } else {
          int64_t NumEdgesFromThisRowAndRelationship =
              cusp_csrs[IdxRelationship].row_offsets[IdxRow + 1] -
              cusp_csrs[IdxRelationship].row_offsets[IdxRow];
          this->row_ptrs[IdxRow + IdxRelationship * this->num_rows + 1] =
              this->row_ptrs[IdxRow + IdxRelationship * this->num_rows] +
              NumEdgesFromThisRowAndRelationship;
          assert(
              this->row_ptrs[IdxRow + IdxRelationship * this->num_rows + 1] ==
              this->row_ptrs[IdxRelationship * this->num_rows] +
                  cusp_csrs[IdxRelationship].row_offsets[IdxRow + 1]);
        }
      }
      assert(this->row_ptrs[(1 + IdxRelationship) * this->num_rows] ==
             my_accumulate<>(num_nnzs.begin(),
                             std::next(num_nnzs.begin(), IdxRelationship + 1),
                             0LL));
      for (int64_t IdxEdgeThisRelationship = 0;
           IdxEdgeThisRelationship < cusp_csrs[IdxRelationship].num_entries;
           IdxEdgeThisRelationship++) {
        this->col_indices[this->row_ptrs[IdxRelationship * this->num_rows] +
                          IdxEdgeThisRelationship] =
            cusp_csrs[IdxRelationship].column_indices[IdxEdgeThisRelationship];
      }
    }
  }

  // template<typename OtherType, typename OtherAlloc>
  void Transpose(/*std::optional<typename std::reference_wrapper<typename thrust::detail::vector_base<OtherType, OtherAlloc>>> eids*/)
    {
    assert(this->num_rels == 1);
    // The TransposeCSR() this function relies on only works for
    // single-relationship CSR. Needs to execute one relationship by one
    // relationship in this heteroseparateCSR case
    assert(num_rows == num_cols);
    for (int64_t idx_relation = 0; idx_relation < this->num_rels;
         idx_relation++) {
      thrust::host_vector<IdxType> row_ptrs_curr_relation(
          this->row_ptrs.begin() + idx_relation * this->num_rows,
          this->row_ptrs.begin() + (idx_relation + 1) * this->num_rows);
      thrust::host_vector<IdxType> col_indices_curr_relation(
          this->col_indices.begin() +
              this->row_ptrs[idx_relation * this->num_rows],
          this->col_indices.begin() +
              this->row_ptrs[(idx_relation + 1) * this->num_rows]);
      thrust::host_vector<IdxType> eids_curr_relation(
          this->eids.begin() + this->row_ptrs[idx_relation * this->num_rows],
          this->eids.begin() +
              this->row_ptrs[(idx_relation + 1) * this->num_rows]);
      thrust::host_vector<IdxType> permutation =
          TransposeCSR(row_ptrs_curr_relation, col_indices_curr_relation);

      thrust::detail::vector_base<IdxType, Alloc> new_eids_curr_relation(
          eids_curr_relation.size());
      typedef typename thrust::detail::vector_base<IdxType, Alloc>::iterator
          ElementIterator;
      typedef typename thrust::host_vector<IdxType>::iterator IndexIterator;
      thrust::permutation_iterator<ElementIterator, IndexIterator> permute_iter(
          eids_curr_relation.begin(), permutation.begin());
      thrust::copy(permute_iter, permute_iter + eids.size(),
                   new_eids_curr_relation.begin());
      thrust::copy(
          new_eids_curr_relation.begin(), new_eids_curr_relation.end(),
          eids.begin() + this->row_ptrs[idx_relation * this->num_rows]);
      thrust::copy(
          row_ptrs_curr_relation.begin(),
          row_ptrs_curr_relation.begin() + row_ptrs_curr_relation.size(),
          this->row_ptrs.begin() + idx_relation * this->num_rows);
      thrust::copy(
          col_indices_curr_relation.begin(),
          col_indices_curr_relation.begin() + col_indices_curr_relation.size(),
          this->col_indices.begin() +
              this->row_ptrs[idx_relation * this->num_rows]);
      //}
    }
  }

  template <typename OtherAlloc>
  MyHeteroSeparateCSR(const MyHeteroSeparateCSR<IdxType, OtherAlloc> &csr) {
    this->row_ptrs = csr.row_ptrs;
    this->col_indices = csr.col_indices;
    this->num_nnzs = csr.num_nnzs;
    this->num_rows = csr.num_rows;
    this->num_cols = csr.num_cols;
    this->num_rels = csr.num_rels;
    this->total_num_nnzs = csr.total_num_nnzs;
    this->eids = csr.eids;
  }
  void VerifyThrustAllocatorRecognizable() const {
    assert(IsHostVector(row_ptrs) || IsDeviceVector(row_ptrs));
  }
  bool IsDataOnGPU() const { return IsDeviceVector(row_ptrs); }
  bool IsDataOnCPU() const { return IsHostVector(row_ptrs); }
  template <typename OtherAlloc>
  std::vector<thrust::detail::vector_base<IdxType, OtherAlloc>>
  GetUniqueSrcNodeIdxForEachRelation() {
    std::vector<thrust::detail::vector_base<IdxType, OtherAlloc>> result;
    for (int64_t IdxRelation = 0; IdxRelation < num_rels; IdxRelation++) {
      thrust::detail::vector_base<thrust::pair<IdxType, IdxType>, Alloc>
          adjacent_rowptr_differences(num_rows + 1);
      thrust::adjacent_difference(
          row_ptrs.begin() + IdxRelation * num_rows,
          row_ptrs.begin() + (IdxRelation + 1) * num_rows + 1,
          adjacent_rowptr_differences.begin());

      // From https://stackoverflow.com/a/29848688/5555077
      // find out the row indices where row_ptr[idx+1]- row_ptr[idx] is non-zero
      thrust::device_vector<int> indices(num_rows);
      thrust::device_vector<int>::iterator end =
          thrust::copy_if(thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(num_rows),
                          std::next(row_ptrs.begin(), 1), indices.begin(),
                          thrust::placeholders::_1 > 0);
      int size = end - indices.begin();
      indices.resize(size);
      result.emplace_back(indices);
    }
    thrust::detail::vector_base<IdxType, Alloc> curr_result;
    return result;
  }
  template <typename OtherAlloc>
  std::vector<thrust::detail::vector_base<IdxType, OtherAlloc>>
  GetUniqueDestNodeIdxForEachRelation() {
    std::vector<thrust::detail::vector_base<IdxType, OtherAlloc>> result;
    for (int64_t IdxRelation = 0; IdxRelation < num_rels; IdxRelation++) {
      thrust::detail::vector_base<IdxType, Alloc>
          dest_node_indices_for_this_relation;
      dest_node_indices_for_this_relation.resize(num_nnzs[IdxRelation]);
      thrust::copy(col_indices.begin() + row_ptrs[IdxRelation],
                   col_indices.begin() + row_ptrs[IdxRelation + 1],
                   dest_node_indices_for_this_relation.begin());
      thrust::sort(dest_node_indices_for_this_relation.begin(),
                   dest_node_indices_for_this_relation.end());
      auto new_end = thrust::unique(dest_node_indices_for_this_relation.begin(),
                                    dest_node_indices_for_this_relation.end());
      result.emplace_back(dest_node_indices_for_this_relation.begin(), new_end);
    }
    thrust::detail::vector_base<IdxType, Alloc> curr_result;
    return result;
  }
};

template <typename IdxType, typename Alloc>
class MyHeteroIntegratedCSR {
 public:
  MyHeteroIntegratedCSR() = default;

  int64_t num_rows;
  int64_t num_cols;
  int64_t num_rels;
  std::vector<int64_t> num_nnzs;
  int64_t total_num_nnzs;
  thrust::detail::vector_base<IdxType, Alloc> row_ptrs;
  thrust::detail::vector_base<IdxType, Alloc> col_indices;
  thrust::detail::vector_base<IdxType, Alloc> rel_type;
  thrust::detail::vector_base<IdxType, Alloc> eids;
  template <typename OtherAlloc>
  MyHeteroIntegratedCSR(
      const int64_t num_rows, const int64_t num_cols, const int64_t num_rels,
      const std::vector<int64_t> &num_nnzs,
      const thrust::detail::vector_base<IdxType, OtherAlloc> &row_ptrs,
      const thrust::detail::vector_base<IdxType, OtherAlloc> &col_indices,
      const thrust::detail::vector_base<IdxType, OtherAlloc> &rel_type,
      const thrust::detail::vector_base<IdxType, OtherAlloc> &eids) {
    this->total_num_nnzs =
        my_accumulate<>(num_nnzs.begin(), num_nnzs.end(), 0LL);
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->num_rels = num_rels;
    this->num_nnzs = num_nnzs;
    this->row_ptrs = row_ptrs;
    this->col_indices = col_indices;
    this->rel_type = rel_type;
    this->eids = eids;
  }

  MyHeteroIntegratedCSR(
      const thrust::detail::vector_base<IdxType, std::allocator<IdxType>>
          &row_ptrs,
      const thrust::detail::vector_base<IdxType, std::allocator<IdxType>>
          &col_indices,
      const thrust::detail::vector_base<IdxType, std::allocator<IdxType>>
          &rel_type,
      const thrust::detail::vector_base<IdxType, std::allocator<IdxType>>
          &eids) {
    this->row_ptrs = row_ptrs;
    this->col_indices = col_indices;
    this->rel_type = rel_type;
    this->num_rows = row_ptrs.size() - 1;
    this->num_cols = this->num_rows;
    // num rels is the largest index in rel_type
    this->num_rels = (*std::max_element(rel_type.begin(), rel_type.end())) + 1;
    this->total_num_nnzs = col_indices.size();
    // count rel_type to get num_nnz of each type
    std::vector<int64_t> num_nnz_type(this->num_rels, 0);
    for (int64_t i = 0; i < this->total_num_nnzs; i++) {
      num_nnz_type[rel_type[i]]++;
    }
    this->num_nnzs = num_nnz_type;
    this->eids = eids;
  }

  template <typename OtherAlloc>
  MyHeteroIntegratedCSR(const MyHeteroIntegratedCSR<IdxType, OtherAlloc> &csr) {
    this->rel_type = csr.rel_type;
    this->row_ptrs = csr.row_ptrs;
    this->col_indices = csr.col_indices;
    this->eids = csr.eids;
    this->num_nnzs = csr.num_nnzs;
    this->num_rows = csr.num_rows;
    this->num_cols = csr.num_cols;
    this->num_rels = csr.num_rels;
    this->total_num_nnzs = csr.total_num_nnzs;
  }

  void Transpose() {
    assert(num_rows == num_cols);

    thrust::host_vector<IdxType> permutation =
        TransposeCSR(row_ptrs, col_indices);

    // work on rel_types
    typedef typename thrust::detail::vector_base<IdxType, Alloc>::iterator
        ElementIterator;
    typedef typename thrust::host_vector<IdxType>::iterator IndexIterator;
    thrust::detail::vector_base<IdxType, Alloc> new_rel_types(
        permutation.size());
    thrust::permutation_iterator<ElementIterator, IndexIterator> permute_iter(
        rel_type.begin(), permutation.begin());
    thrust::copy(permute_iter, permute_iter + permutation.size(),
                 new_rel_types.begin());

    thrust::copy(new_rel_types.begin(), new_rel_types.end(), rel_type.begin());

    // work on eids
    thrust::detail::vector_base<IdxType, Alloc> new_eids(permutation.size());
    thrust::permutation_iterator<ElementIterator, IndexIterator>
        permute_iter_eids(this->eids.begin(), permutation.begin());
    thrust::copy(permute_iter_eids, permute_iter_eids + permutation.size(),
                 new_eids.begin());
    thrust::copy(new_eids.begin(), new_eids.end(), this->eids.begin());
  }

  bool IsSortedByEdgeType_CPU() {
    assert(IsDataOnCPU());
    for (int64_t IdxRow = 0; IdxRow < num_rows; IdxRow++) {
      std::vector<std::pair<IdxType, IdxType>> EdgeRelationshipPairFromThisNode;
      for (IdxType IdxEdge = row_ptrs[IdxRow]; IdxEdge < row_ptrs[IdxRow + 1];
           IdxEdge++) {
        IdxType IdxSrcNode = IdxRow;
        IdxType IdxDestNode = col_indices[IdxEdge];
        IdxType IdxRelationshipEdge = rel_type[IdxEdge];
        EdgeRelationshipPairFromThisNode.push_back(
            std::make_pair(IdxDestNode, IdxRelationshipEdge));
      }
      std::sort(EdgeRelationshipPairFromThisNode.begin(),
                EdgeRelationshipPairFromThisNode.end(),
                [](const std::pair<IdxType, IdxType> &a,
                   const std::pair<IdxType, IdxType> &b) {
                  return a.second < b.second;
                });

      // check if the input csr is sorted
      for (int64_t IdxEdge = 0;
           IdxEdge < EdgeRelationshipPairFromThisNode.size(); IdxEdge++) {
        if (col_indices[row_ptrs[IdxRow] + IdxEdge] !=
            EdgeRelationshipPairFromThisNode[IdxEdge].first) {
          return false;
        }

        if (rel_type[row_ptrs[IdxRow] + IdxEdge] =
                EdgeRelationshipPairFromThisNode[IdxEdge].second) {
          return false;
        }
      }
    }
    return true;
  }

  void SortByEdgeType_CPU(
      /*const thrust::detail::vector_base<IdxType, Alloc>& eids*/) {
    assert(IsDataOnCPU());
    for (int64_t IdxRow = 0; IdxRow < num_rows; IdxRow++) {
      std::vector<std::pair<std::pair<IdxType, IdxType>, IdxType>>
          EdgeRelationshipEidsTupleFromThisNode;
      for (IdxType IdxEdge = row_ptrs[IdxRow]; IdxEdge < row_ptrs[IdxRow + 1];
           IdxEdge++) {
        IdxType IdxSrcNode = IdxRow;
        IdxType IdxDestNode = col_indices[IdxEdge];
        IdxType IdxRelationshipEdge = rel_type[IdxEdge];
        IdxType ElementEids = eids[IdxEdge];
        EdgeRelationshipEidsTupleFromThisNode.push_back(std::make_pair(
            std::make_pair(IdxDestNode, IdxRelationshipEdge), ElementEids));
      }
      std::sort(EdgeRelationshipEidsTupleFromThisNode.begin(),
                EdgeRelationshipEidsTupleFromThisNode.end(),
                [](const std::pair<IdxType, IdxType> &a,
                   const std::pair<IdxType, IdxType> &b) {
                  return a.first.second < b.first.second;
                });
      // write back
      for (int64_t IdxEdge = 0;
           IdxEdge < EdgeRelationshipEidsTupleFromThisNode.size(); IdxEdge++) {
        col_indices[row_ptrs[IdxRow] + IdxEdge] =
            EdgeRelationshipEidsTupleFromThisNode[IdxEdge].first.first;
        rel_type[row_ptrs[IdxRow] + IdxEdge] =
            EdgeRelationshipEidsTupleFromThisNode[IdxEdge].first.second;
        eids[row_ptrs[IdxRow] + IdxEdge] =
            EdgeRelationshipEidsTupleFromThisNode[IdxEdge].second;
      }
    }
  }

  void VerifyThrustAllocatorRecognizable() const {
    // the allocator should be either std::allocator (thrust::host_vector's) or
    // thrust::device_allocator (thrust::device_vector's)
    assert(IsHostVector(rel_type) || IsDeviceVector(rel_type));
  }

  bool IsDataOnGPU() const { return IsDeviceVector(rel_type); }
  bool IsDataOnCPU() const { return IsHostVector(rel_type); }
};

template <typename IdxType, typename Alloc, typename CSRType>
class MySegmentCSR {
 public:
  int64_t NonCutoffMaxNodeIndex;
  int64_t num_rows;
  int64_t num_cols;
  int64_t num_rels;
  std::vector<int64_t> num_nnzs;
  std::vector<int64_t> exclusive_scan_num_nnzs;
  std::vector<int64_t> dense_num_nnzs;
  std::vector<int64_t> exclusive_scan_dense_num_nnzs;
  int64_t dense_total_num_nnzs;
  int64_t total_num_nnzs;
  CSRType csr;
  thrust::detail::vector_base<IdxType, Alloc> maximal_edge_num_per_src_node;
  thrust::detail::vector_base<IdxType, Alloc>
      padded_exclusive_scan_maximal_edge_num_per_src_node;
  thrust::detail::vector_base<IdxType, Alloc> maximal_edge_type_per_src_node;
  thrust::detail::vector_base<IdxType, Alloc> src_node_per_edge_type;
  thrust::detail::vector_base<IdxType, Alloc> num_src_nodes_per_edge_type;
  thrust::detail::vector_base<IdxType, Alloc>
      exclusive_scan_num_src_nodes_per_edge_type;
  thrust::detail::vector_base<IdxType, Alloc> padded_dense_edges;
  thrust::detail::vector_base<IdxType, Alloc> padded_dense_edges_eids;

  template <typename OtherAlloc, typename OtherAlloc2>
  MySegmentCSR(const int64_t num_rows, const int64_t num_cols,
               const int64_t maximal_non_cutoff_node_indices,
               const thrust::detail::vector_base<IdxType, OtherAlloc>
                   &maximal_edge_num_per_src_node,
               const thrust::detail::vector_base<IdxType, OtherAlloc>
                   &maximal_edge_type_per_src_node,
               const thrust::detail::vector_base<IdxType, OtherAlloc>
                   &src_node_per_edge_type,
               const thrust::detail::vector_base<IdxType, OtherAlloc>
                   &num_src_nodes_per_edge_type,
               const thrust::detail::vector_base<IdxType, OtherAlloc2>
                   &padded_dense_edges,
               const thrust::detail::vector_base<IdxType, OtherAlloc2>
                   &padded_exclusive_scan_maximal_edge_num_per_src_node,
               const CSRType &residue_coo_matrix_h,
               const thrust::detail::vector_base<IdxType, OtherAlloc2>
                   &padded_dense_edges_eids) {
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->num_rels = num_src_nodes_per_edge_type.size();
    this->NonCutoffMaxNodeIndex = maximal_non_cutoff_node_indices;
    this->csr = residue_coo_matrix_h;
    this->maximal_edge_num_per_src_node = maximal_edge_num_per_src_node;
    this->maximal_edge_type_per_src_node = maximal_edge_type_per_src_node;
    this->src_node_per_edge_type = src_node_per_edge_type;
    this->num_src_nodes_per_edge_type = num_src_nodes_per_edge_type;
    this->padded_dense_edges = padded_dense_edges;
    this->padded_exclusive_scan_maximal_edge_num_per_src_node =
        padded_exclusive_scan_maximal_edge_num_per_src_node;
    this->num_nnzs.resize(this->num_rels);
    this->padded_dense_edges_eids = padded_dense_edges_eids;

    // count dense region num nnzs and total_num_nnzs
    dense_num_nnzs.resize(this->num_rels);
    num_nnzs.resize(this->num_rels);
    for (int IdxRelationship = 0; IdxRelationship < num_nnzs.size();
         IdxRelationship++) {
      dense_num_nnzs[IdxRelationship] = 0;
      // iterates source node with the maximal edge type and count the number of
      // edges there
      for (int IdxSourceNode = 0;
           IdxSourceNode < num_src_nodes_per_edge_type[IdxRelationship];
           IdxSourceNode++) {
        dense_num_nnzs[IdxRelationship] +=
            maximal_edge_num_per_src_node[IdxSourceNode];
      }
      this->num_nnzs[IdxRelationship] =
          dense_num_nnzs[IdxRelationship] + this->csr.num_nnzs[IdxRelationship];
    }

    // exclusive scan
    this->exclusive_scan_dense_num_nnzs.resize(this->num_rels + 1);
    this->exclusive_scan_num_nnzs.resize(this->num_rels + 1);
    this->exclusive_scan_num_nnzs[0] = 0;
    this->exclusive_scan_dense_num_nnzs[0] = 0;
    for (int IdxRelationship = 0; IdxRelationship < this->num_rels;
         IdxRelationship++) {
      this->exclusive_scan_num_nnzs[IdxRelationship + 1] =
          this->exclusive_scan_num_nnzs[IdxRelationship] +
          this->num_nnzs[IdxRelationship];
      this->exclusive_scan_dense_num_nnzs[IdxRelationship + 1] =
          this->exclusive_scan_dense_num_nnzs[IdxRelationship] +
          dense_num_nnzs[IdxRelationship];
    }

    this->exclusive_scan_num_src_nodes_per_edge_type.resize(this->num_rels + 1);
    this->exclusive_scan_num_src_nodes_per_edge_type[0] = 0;
    for (int IdxRelationship = 0; IdxRelationship < this->num_rels;
         IdxRelationship++) {
      this->exclusive_scan_num_src_nodes_per_edge_type[IdxRelationship + 1] =
          this->exclusive_scan_num_src_nodes_per_edge_type[IdxRelationship] +
          num_src_nodes_per_edge_type[IdxRelationship];
    }

    this->dense_total_num_nnzs =
        my_accumulate<>(this->num_nnzs.begin(), this->num_nnzs.end(), 0LL);
    this->total_num_nnzs =
        this->dense_total_num_nnzs + this->csr.total_num_nnzs;
  }

  template <typename OtherAlloc, typename OtherCSRType>
  MySegmentCSR(const MySegmentCSR<IdxType, OtherAlloc, OtherCSRType> &other) {
    this->csr = other.csr;
    this->num_rows = other.num_rows;
    this->num_cols = other.num_cols;
    this->num_rels = other.num_rels;
    this->NonCutoffMaxNodeIndex = other.NonCutoffMaxNodeIndex;
    this->num_nnzs = other.num_nnzs;
    this->dense_num_nnzs = other.dense_num_nnzs;
    this->exclusive_scan_num_nnzs = other.exclusive_scan_num_nnzs;
    this->exclusive_scan_dense_num_nnzs = other.exclusive_scan_dense_num_nnzs;
    this->dense_total_num_nnzs = other.dense_total_num_nnzs;
    this->total_num_nnzs = other.total_num_nnzs;
    this->maximal_edge_num_per_src_node = other.maximal_edge_num_per_src_node;
    this->maximal_edge_type_per_src_node = other.maximal_edge_type_per_src_node;
    this->src_node_per_edge_type = other.src_node_per_edge_type;
    this->num_src_nodes_per_edge_type = other.num_src_nodes_per_edge_type;
    this->exclusive_scan_num_src_nodes_per_edge_type =
        other.exclusive_scan_num_src_nodes_per_edge_type;
    this->padded_dense_edges = other.padded_dense_edges;
    this->padded_exclusive_scan_maximal_edge_num_per_src_node =
        other.padded_exclusive_scan_maximal_edge_num_per_src_node;
    this->padded_dense_edges_eids = other.padded_dense_edges_eids;
  }
};

template <typename Idx>
Idx my_ceil_div(const Idx a, const Idx b) {
  return (a + b - 1) / b;
}

template <typename Idx, typename Alloc>
std::pair<thrust::host_vector<Idx>, thrust::host_vector<Idx>>
MySegmentCSRPadDenseEdges(
    const thrust::detail::vector_base<Idx, Alloc> &dense_edges,
    const thrust::detail::vector_base<Idx, Alloc> &dense_edges_num_per_src_node,
    const int padding_factor) {
  int64_t num_rows = dense_edges_num_per_src_node.size();
  thrust::host_vector<Idx> padded_exclusive_scan_dense_edge_num_per_src_node(
      dense_edges_num_per_src_node.size() + 1);
  thrust::host_vector<Idx> exclusive_scan_dense_edge_num_per_src_node(
      dense_edges_num_per_src_node.size() + 1);
  thrust::host_vector<Idx> padded_dense_edges;
  // Each element idx in this array serves as the starting position of of source
  // node idx in dense_edges
  padded_exclusive_scan_dense_edge_num_per_src_node[0] = 0;
  exclusive_scan_dense_edge_num_per_src_node[0] = 0;
  for (int64_t IdxSourceNode = 0; IdxSourceNode < num_rows; IdxSourceNode++) {
    int64_t num_edges_for_curr_node =
        dense_edges_num_per_src_node[IdxSourceNode];
    int64_t padded_num_edges_for_curr_node =
        my_ceil_div<int>(num_edges_for_curr_node, padding_factor) *
        padding_factor;
    int64_t starting_position_of_curr_node =
        exclusive_scan_dense_edge_num_per_src_node[IdxSourceNode];
    for (int64_t IdxEdge = 0; IdxEdge < num_edges_for_curr_node; IdxEdge++) {
      padded_dense_edges.push_back(
          dense_edges[starting_position_of_curr_node + IdxEdge]);
    }
    for (int64_t IdxPaddingElement = 0;
         IdxPaddingElement <
         padded_num_edges_for_curr_node - num_edges_for_curr_node;
         IdxPaddingElement++) {
      padded_dense_edges.push_back(MyHyb_NONEXISTENT_ELEMENT);
    }
    padded_exclusive_scan_dense_edge_num_per_src_node[IdxSourceNode + 1] =
        padded_exclusive_scan_dense_edge_num_per_src_node[IdxSourceNode] +
        padded_num_edges_for_curr_node;
    exclusive_scan_dense_edge_num_per_src_node[IdxSourceNode + 1] =
        exclusive_scan_dense_edge_num_per_src_node[IdxSourceNode] +
        num_edges_for_curr_node;
    assert(
        padded_exclusive_scan_dense_edge_num_per_src_node[IdxSourceNode + 1] ==
        padded_dense_edges.size());
  }
  return std::make_pair(padded_exclusive_scan_dense_edge_num_per_src_node,
                        padded_dense_edges);
}

// TODO: implement AOS for eids, rel_types and col_indices
template <typename IdxType, typename Alloc, typename CSRType>
class MyHyb {
  //[0,HybIndexMax] has both ELL and CSR format
  //(HybIndexMax, num_rows) has only CSR format
 public:
  int64_t HybIndexMax;
  int64_t ELL_logical_width;
  int64_t ELL_physical_width;  // logical width and actual width is similar to
                               // lda in BLAS routines.
  int64_t num_rows;
  int64_t num_cols;
  int64_t num_rels;
  std::vector<int64_t> num_nnzs;
  int64_t total_num_nnzs;
  CSRType csr;
  thrust::detail::vector_base<IdxType, Alloc> ELLColIdx;
  thrust::detail::vector_base<IdxType, Alloc> ELLRelType;
  thrust::detail::vector_base<IdxType, Alloc> ELLEids;

  MyHyb() = default;

  MyHyb(const int64_t HybIndexMax, const int64_t ELL_logical_width,
        const int64_t ELL_physical_width, const CSRType &csr,
        const thrust::detail::vector_base<IdxType, Alloc> &ELLColIdx,
        const thrust::detail::vector_base<IdxType, Alloc> &ELLRelType,
        const thrust::detail::vector_base<IdxType, Alloc> &ELLEids,
        const int64_t num_rows, const int64_t num_cols, const int64_t num_rels,
        const std::vector<int64_t> &num_nnzs) {
    this->HybIndexMax = HybIndexMax;
    this->ELL_logical_width = ELL_logical_width;
    this->ELL_physical_width = ELL_physical_width;
    this->csr = csr;
    this->ELLColIdx = ELLColIdx;
    this->ELLRelType = ELLRelType;
    this->ELLEids = ELLEids;
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->num_rels = num_rels;
    this->num_nnzs = num_nnzs;
    this->total_num_nnzs =
        my_accumulate<>(num_nnzs.begin(), num_nnzs.end(), 0LL);
  }

  template <typename OtherAlloc, typename OtherCSRType>
  MyHyb(const MyHyb<IdxType, OtherAlloc, OtherCSRType> &another_myhyb) {
    this->HybIndexMax = another_myhyb.HybIndexMax;
    this->ELL_logical_width = another_myhyb.ELL_logical_width;
    this->ELL_physical_width = another_myhyb.ELL_physical_width;
    this->csr = another_myhyb.csr;
    this->ELLColIdx = another_myhyb.ELLColIdx;
    this->ELLRelType = another_myhyb.ELLRelType;
    this->ELLEids = another_myhyb.ELLEids;
    this->num_rows = another_myhyb.num_rows;
    this->num_cols = another_myhyb.num_cols;
    this->num_rels = another_myhyb.num_rels;
    this->num_nnzs = another_myhyb.num_nnzs;
    this->total_num_nnzs = another_myhyb.total_num_nnzs;
  }

  void VerifyThrustAllocatorRecognizable() const {
    // the allocator should be either std::allocator (thrust::host_vector's) or
    // thrust::device_allocator (thrust::device_vector's)
    assert(IsHostVector(ELLRelType) || IsDeviceVector(ELLRelType));
  }

  bool IsDataOnGPU() const { return IsDeviceVector(ELLRelType); }
  bool IsDataOnCPU() const { return IsHostVector(ELLRelType); }
};

// separate CSR to integrated CSR
template <typename IdxType>
// MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>>
MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>>
ToIntegratedCSR_GPU(
    const MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>
        &csr) {
  thrust::device_vector<IdxType> result_row_ptrs;
  thrust::device_vector<IdxType> result_col_indices;
  thrust::device_vector<IdxType> result_rel_type;
  thrust::device_vector<IdxType> result_eids;
  // TODO: implement here
  assert(0 && "GPU kernel not implemented");
  MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>> result(
      csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptrs,
      result_col_indices, result_rel_type, result_eids);
  // return dummy to avoid compiler error
  return result;
}

template <typename IdxType>
MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> ToIntegratedCSR_CPU(
    const MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> &csr) {
  int64_t total_num_nnzs =
      my_accumulate<>(csr.num_nnzs.begin(), csr.num_nnzs.end(), 0LL);

  std::vector<std::map<IdxType, std::vector<std::pair<IdxType, IdxType>>>>
      edge_to_rel_type_and_eid(csr.num_rows);

  thrust::host_vector<IdxType> result_row_ptrs(csr.num_rows + 1, 0);
  thrust::host_vector<IdxType> result_col_indices(total_num_nnzs, 0);
  thrust::host_vector<IdxType> result_rel_type(total_num_nnzs, 0);
  thrust::host_vector<IdxType> result_eids(total_num_nnzs, 0);

  for (int64_t IdxRelation = 0; IdxRelation < csr.num_rels; IdxRelation++) {
    for (int64_t IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
      for (IdxType IdxElementColIdxWithOffset =
               csr.row_ptrs[IdxRelation * csr.num_rows + IdxRow];
           IdxElementColIdxWithOffset <
           csr.row_ptrs[IdxRelation * csr.num_rows + IdxRow + 1];
           IdxElementColIdxWithOffset++) {
        IdxType IdxSrcNode = IdxRow;
        IdxType IdxDestNode = csr.col_indices[IdxElementColIdxWithOffset];
        IdxType IdxRelationshipEdge = IdxRelation;
        IdxType IdxEdge = csr.eids[IdxElementColIdxWithOffset];
        edge_to_rel_type_and_eid[IdxSrcNode][IdxDestNode].push_back(
            std::make_pair(IdxRelationshipEdge, IdxEdge));
      }
    }
  }

  IdxType currEdgeIdx = 0;
  for (int64_t IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
    for (auto it = edge_to_rel_type_and_eid[IdxRow].begin();
         it != edge_to_rel_type_and_eid[IdxRow].end(); it++) {
      for (IdxType IdxElement = 0; IdxElement < it->second.size();
           IdxElement++) {
        result_col_indices[currEdgeIdx] = it->first;
        result_rel_type[currEdgeIdx] = it->second[IdxElement].first;
        result_eids[currEdgeIdx] = it->second[IdxElement].second;
        currEdgeIdx += 1;
        result_row_ptrs[IdxRow + 1] += 1;
      }
    }
    result_row_ptrs[IdxRow + 1] =
        result_row_ptrs[IdxRow] + result_row_ptrs[IdxRow + 1];
    assert(result_row_ptrs[IdxRow + 1] == currEdgeIdx);
  }

  MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> result(
      csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptrs,
      result_col_indices, result_rel_type, result_eids);

  return result;
}

// integrated CSR to separate CSR
template <typename IdxType>
// MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>
MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>
ToSeparateCSR_GPU(
    const MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>>
        &csr) {
  thrust::device_vector<IdxType> result_row_ptrs;
  thrust::device_vector<IdxType> result_col_indices;
  thrust::device_vector<IdxType> result_eids;
  // TODO: implement here
  assert(0 && "GPU kernel not implemented");

  MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>> result(
      csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptrs,
      result_col_indices, result_eids);
  // return dummy to avoid compiler error
  return result;
}

template <typename IdxType>
bool IsEqual(
    const MyHeteroIntegratedCSR<
        IdxType, typename thrust::host_vector<IdxType>::allocator_type> &csr1,
    const MyHeteroIntegratedCSR<
        IdxType, typename thrust::host_vector<IdxType>::allocator_type> &csr2) {
  if (csr1.num_rows != csr2.num_rows) {
    return false;
  }
  if (csr1.num_cols != csr2.num_cols) {
    return false;
  }
  if (csr1.num_rels != csr2.num_rels) {
    return false;
  }
  if (csr1.total_num_nnzs != csr2.total_num_nnzs) {
    return false;
  }
  if (csr1.num_nnzs.size() != csr2.num_nnzs.size()) {
    return false;
  }
  for (int64_t idx_relationship = 0; idx_relationship < csr1.num_nnzs.size();
       idx_relationship++) {
    if (csr1.num_nnzs[idx_relationship] != csr2.num_nnzs[idx_relationship]) {
      return false;
    }
  }

  for (IdxType IdxRow = 0; IdxRow < csr1.num_rows; IdxRow++) {
    if (csr1.row_ptrs[IdxRow + 1] != csr2.row_ptrs[IdxRow + 1]) {
      return false;
    }
  }

  for (int64_t IdxRow = 0; IdxRow < csr1.num_rows; IdxRow++) {
    std::set<std::pair<IdxType, std::pair<IdxType, IdxType>>>
        DestNodeRelationshipEIDsTriplets;
    for (int64_t IdxEdge = csr1.row_ptrs[IdxRow];
         IdxEdge < csr1.row_ptrs[IdxRow + 1]; IdxEdge++) {
      DestNodeRelationshipEIDsTriplets.insert(std::make_pair(
          csr1.col_indices[IdxEdge],
          std::make_pair(csr1.rel_type[IdxEdge], csr1.eids[IdxEdge])));
      DestNodeRelationshipEIDsTriplets.insert(std::make_pair(
          csr2.col_indices[IdxEdge],
          std::make_pair(csr2.rel_type[IdxEdge], csr2.eids[IdxEdge])));
    }
    // if there is any difference between the two CSRs, the size of the set will
    // be larger than the difference of one csr row ptr
    if (DestNodeRelationshipEIDsTriplets.size() !=
        csr2.row_ptrs[IdxRow + 1] - csr2.row_ptrs[IdxRow]) {
      return false;
    }
  }
  return true;
}

template <typename IdxType>
bool IsEqual(
    const MyHeteroSeparateCSR<
        IdxType, typename thrust::host_vector<IdxType>::allocator_type> &csr1,
    const MyHeteroSeparateCSR<
        IdxType, typename thrust::host_vector<IdxType>::allocator_type> &csr2) {
  if (csr1.num_rows != csr2.num_rows) {
    return false;
  }
  if (csr1.num_cols != csr2.num_cols) {
    return false;
  }
  if (csr1.num_rels != csr2.num_rels) {
    return false;
  }
  if (csr1.total_num_nnzs != csr2.total_num_nnzs) {
    return false;
  }
  if (csr1.num_nnzs.size() != csr2.num_nnzs.size()) {
    return false;
  }
  for (int64_t idx_relationship = 0; idx_relationship < csr1.num_nnzs.size();
       idx_relationship++) {
    if (csr1.num_nnzs[idx_relationship] != csr2.num_nnzs[idx_relationship]) {
      return false;
    }
  }

  for (int64_t IdxRelationship = 0; IdxRelationship < csr1.num_rels;
       IdxRelationship++) {
    for (int64_t IdxRow = 0; IdxRow < csr1.num_rows; IdxRow++) {
      if (csr1.row_ptrs[IdxRelationship * csr1.num_rows + IdxRow + 1] !=
          csr2.row_ptrs[IdxRelationship * csr1.num_rows + IdxRow + 1]) {
        return false;
      }
    }
  }

  for (int64_t IdxRowWithRelationship = 0;
       IdxRowWithRelationship < csr1.num_rows * csr1.num_rels;
       IdxRowWithRelationship++) {
    std::set<std::pair<IdxType, IdxType>> DestNodeWithEids;
    for (int64_t IdxEdge = csr1.row_ptrs[IdxRowWithRelationship];
         IdxEdge < csr1.row_ptrs[IdxRowWithRelationship + 1]; IdxEdge++) {
      DestNodeWithEids.insert(
          std::make_pair(csr1.col_indices[IdxEdge], csr1.eids[IdxEdge]));
      DestNodeWithEids.insert(
          std::make_pair(csr2.col_indices[IdxEdge], csr2.eids[IdxEdge]));
    }
    // if there is any difference between the two CSRs, the size of the set will
    // be larger than the difference of one csr row ptr
    if (DestNodeWithEids.size() != csr2.row_ptrs[IdxRowWithRelationship + 1] -
                                       csr2.row_ptrs[IdxRowWithRelationship]) {
      return false;
    }
  }
  return true;
}

template <typename IdxType>
MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> ToSeparateCSR_CPU(
    const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> &csr) {
  thrust::host_vector<IdxType> result_row_ptrs(csr.num_rows * csr.num_rels + 1,
                                               0);
  thrust::host_vector<IdxType> result_col_indices(csr.total_num_nnzs, 0);
  thrust::host_vector<IdxType> result_eids(csr.total_num_nnzs, 0);
  std::vector<std::vector<std::vector<std::pair<IdxType, IdxType>>>>
      rel_type_to_edges(
          csr.num_rels,
          std::vector<std::vector<std::pair<IdxType, IdxType>>>(
              csr.num_rows, std::vector<std::pair<IdxType, IdxType>>()));

  for (int64_t IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
    for (IdxType IdxEdge = csr.row_ptrs[IdxRow];
         IdxEdge < csr.row_ptrs[IdxRow + 1]; IdxEdge++) {
      IdxType IdxSrcNode = IdxRow;
      IdxType IdxDestNode = csr.col_indices[IdxEdge];
      IdxType IdxRelationshipEdge = csr.rel_type[IdxEdge];
      IdxType IdxEids = csr.eids[IdxEdge];
      rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(
          std::make_pair(IdxDestNode, IdxEids));
    }
  }
  // result_row_ptr stores the absolute offset from base address of the whole
  // col_idx array, rather than relative offset per relation
  for (int64_t IdxRelationship = 0; IdxRelationship < csr.num_rels;
       IdxRelationship++) {
    for (IdxType IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
      result_row_ptrs[IdxRelationship * csr.num_rows + IdxRow + 1] =
          result_row_ptrs[IdxRelationship * csr.num_rows + IdxRow];
      for (IdxType IdxElement = 0;
           IdxElement < rel_type_to_edges[IdxRelationship][IdxRow].size();
           IdxElement++) {
        result_col_indices[result_row_ptrs[IdxRelationship * csr.num_rows +
                                           IdxRow + 1]] =
            rel_type_to_edges[IdxRelationship][IdxRow][IdxElement].first;
        result_eids[result_row_ptrs[IdxRelationship * csr.num_rows + IdxRow +
                                    1]] =
            rel_type_to_edges[IdxRelationship][IdxRow][IdxElement].second;
        result_row_ptrs[IdxRelationship * csr.num_rows + IdxRow + 1] += 1;
      }
    }
  }

  MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> result(
      csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptrs,
      result_col_indices, result_eids);
  return result;
}

template <typename IdxType>
std::vector<thrust::host_vector<IdxType>> ConvertIntegratedCOOToSeparateCSR(
    thrust::host_vector<IdxType> integrated_row_indices,
    thrust::host_vector<IdxType> integrated_col_indices,
    thrust::host_vector<IdxType> integrated_rel_type,
    thrust::host_vector<IdxType> integrated_eids, int64_t num_rows,
    int64_t num_rels) {
  thrust::host_vector<IdxType> result_rel_ptrs(num_rels + 1, 0);
  thrust::host_vector<IdxType> result_row_ptrs(num_rows * num_rels + 1, 0);
  thrust::host_vector<IdxType> result_col_indices(integrated_col_indices.size(),
                                                  0);
  thrust::host_vector<IdxType> result_eids(integrated_col_indices.size(), 0);
  std::vector<std::vector<std::vector<std::pair<IdxType, IdxType>>>>
      rel_type_to_edges(
          num_rels, std::vector<std::vector<std::pair<IdxType, IdxType>>>(
                        num_rows, std::vector<std::pair<IdxType, IdxType>>()));

  for (IdxType IdxEdge = 0; IdxEdge < integrated_col_indices.size();
       IdxEdge++) {
    IdxType IdxSrcNode = integrated_row_indices[IdxEdge];
    IdxType IdxDestNode = integrated_col_indices[IdxEdge];
    IdxType IdxRelationshipEdge = integrated_rel_type[IdxEdge];
    IdxType IdxEids = integrated_eids[IdxEdge];
    rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(
        std::make_pair(IdxDestNode, IdxEids));
  }

  for (int64_t IdxRelationship = 0; IdxRelationship < num_rels;
       IdxRelationship++) {
    for (IdxType IdxRow = 0; IdxRow < num_rows; IdxRow++) {
      result_row_ptrs[IdxRelationship * num_rows + IdxRow + 1] =
          result_row_ptrs[IdxRelationship * num_rows + IdxRow];
      for (IdxType IdxElement = 0;
           IdxElement < rel_type_to_edges[IdxRelationship][IdxRow].size();
           IdxElement++) {
        result_col_indices[result_row_ptrs[IdxRelationship * num_rows + IdxRow +
                                           1]] =
            rel_type_to_edges[IdxRelationship][IdxRow][IdxElement].first;
        result_eids[result_row_ptrs[IdxRelationship * num_rows + IdxRow + 1]] =
            rel_type_to_edges[IdxRelationship][IdxRow][IdxElement].second;
        result_row_ptrs[IdxRelationship * num_rows + IdxRow + 1] += 1;
      }
    }
    result_rel_ptrs[IdxRelationship + 1] =
        result_rel_ptrs[IdxRelationship] +
        result_row_ptrs[IdxRelationship * num_rows + num_rows];
  }

  return {result_rel_ptrs, result_row_ptrs, result_col_indices, result_eids};
}

template <typename IdxType>
std::vector<thrust::host_vector<IdxType>> ConvertIntegratedCOOToSeparateCOO(
    thrust::host_vector<IdxType> integrated_row_indices,
    thrust::host_vector<IdxType> integrated_col_indices,
    thrust::host_vector<IdxType> integrated_rel_type,
    thrust::host_vector<IdxType> integrated_eids, int64_t num_rows,
    int64_t num_rels) {
  thrust::host_vector<IdxType> result_rel_ptrs(num_rels + 1, 0);
  thrust::host_vector<IdxType> result_row_indices(integrated_row_indices.size(),
                                                  0);
  thrust::host_vector<IdxType> result_col_indices(integrated_row_indices.size(),
                                                  0);
  thrust::host_vector<IdxType> result_eids(integrated_row_indices.size(), 0);
  std::vector<std::vector<std::vector<std::pair<IdxType, IdxType>>>>
      rel_type_to_edges(
          num_rels, std::vector<std::vector<std::pair<IdxType, IdxType>>>(
                        num_rows, std::vector<std::pair<IdxType, IdxType>>()));

  for (IdxType IdxEdge = 0; IdxEdge < integrated_row_indices.size();
       IdxEdge++) {
    IdxType IdxSrcNode = integrated_row_indices[IdxEdge];
    IdxType IdxDestNode = integrated_col_indices[IdxEdge];
    IdxType IdxRelationshipEdge = integrated_rel_type[IdxEdge];
    IdxType IdxEids = integrated_eids[IdxEdge];
    rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(
        std::make_pair(IdxDestNode, IdxEids));
  }

  result_rel_ptrs[0] = 0;
  for (int64_t IdxRelationship = 0; IdxRelationship < num_rels;
       IdxRelationship++) {
    result_rel_ptrs[IdxRelationship + 1] = result_rel_ptrs[IdxRelationship];
    for (IdxType IdxRow = 0; IdxRow < num_rows; IdxRow++) {
      for (IdxType IdxElement = 0;
           IdxElement < rel_type_to_edges[IdxRelationship][IdxRow].size();
           IdxElement++) {
        result_row_indices[result_rel_ptrs[IdxRelationship + 1]] = IdxRow;
        result_col_indices[result_rel_ptrs[IdxRelationship + 1]] =
            rel_type_to_edges[IdxRelationship][IdxRow][IdxElement].first;
        result_eids[result_rel_ptrs[IdxRelationship + 1]] =
            rel_type_to_edges[IdxRelationship][IdxRow][IdxElement].second;
        result_rel_ptrs[IdxRelationship + 1] += 1;
      }
    }
  }

  std::vector<thrust::host_vector<IdxType>> result = {
      result_rel_ptrs, result_row_indices, result_col_indices, result_eids};
  return result;
}

// TODO: use template to merge with ToSeparateCSR_CPU
template <typename IdxType>
std::vector<thrust::host_vector<IdxType>> ToSeparateCOO(
    const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> &csr) {
  thrust::host_vector<IdxType> result_rel_ptrs(csr.num_rels + 1, 0);
  thrust::host_vector<IdxType> result_row_indices(csr.total_num_nnzs, 0);
  thrust::host_vector<IdxType> result_col_indices(csr.total_num_nnzs, 0);
  thrust::host_vector<IdxType> result_eids(csr.total_num_nnzs, 0);
  std::vector<std::vector<std::vector<std::pair<IdxType, IdxType>>>>
      rel_type_to_edges(
          csr.num_rels,
          std::vector<std::vector<std::pair<IdxType, IdxType>>>(
              csr.num_rows, std::vector<std::pair<IdxType, IdxType>>()));

  for (int64_t IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
    for (IdxType IdxEdge = csr.row_ptrs[IdxRow];
         IdxEdge < csr.row_ptrs[IdxRow + 1]; IdxEdge++) {
      IdxType IdxSrcNode = IdxRow;
      IdxType IdxDestNode = csr.col_indices[IdxEdge];
      IdxType IdxRelationshipEdge = csr.rel_type[IdxEdge];
      IdxType IdxEids = csr.eids[IdxEdge];
      rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(
          std::make_pair(IdxDestNode, IdxEids));
    }
  }
  result_rel_ptrs[0] = 0;
  for (int64_t IdxRelationship = 0; IdxRelationship < csr.num_rels;
       IdxRelationship++) {
    result_rel_ptrs[IdxRelationship + 1] =
        result_rel_ptrs[IdxRelationship] + csr.num_nnzs[IdxRelationship];
    int64_t curr_relationship_edge_count = 0;
    for (IdxType IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
      for (IdxType IdxElement = 0;
           IdxElement < rel_type_to_edges[IdxRelationship][IdxRow].size();
           IdxElement++) {
        result_row_indices[curr_relationship_edge_count +
                           result_rel_ptrs[IdxRelationship]] = IdxRow;
        result_col_indices[curr_relationship_edge_count +
                           result_rel_ptrs[IdxRelationship]] =
            rel_type_to_edges[IdxRelationship][IdxRow][IdxElement].first;
        result_eids[curr_relationship_edge_count +
                    result_rel_ptrs[IdxRelationship]] =
            rel_type_to_edges[IdxRelationship][IdxRow][IdxElement].second;

        curr_relationship_edge_count++;
      }
    }
  }

  std::vector<thrust::host_vector<IdxType>> result = {
      result_rel_ptrs, result_row_indices, result_col_indices, result_eids};
  return result;
}

template <typename IdxType, typename CSRType>
bool IsEqual(const MyHyb<IdxType, std::allocator<IdxType>, CSRType> &myhyb1,
             const MyHyb<IdxType, std::allocator<IdxType>, CSRType> &myhyb2) {
  // TODO: update with eids
  if (myhyb1.num_rows != myhyb2.num_rows) {
    return false;
  }
  if (myhyb1.num_cols != myhyb2.num_cols) {
    return false;
  }
  if (myhyb1.num_rels != myhyb2.num_rels) {
    return false;
  }
  if (myhyb1.total_num_nnzs != myhyb2.total_num_nnzs) {
    return false;
  }
  if (myhyb1.num_nnzs.size() != myhyb2.num_nnzs.size()) {
    return false;
  }
  for (int64_t IdxRelationship = 0; IdxRelationship < myhyb1.num_rels;
       IdxRelationship++) {
    if (myhyb1.num_nnzs[IdxRelationship] != myhyb2.num_nnzs[IdxRelationship]) {
      return false;
    }
  }
  if (myhyb1.HybIndexMax != myhyb2.HybIndexMax) {
    return false;
  }
  if (myhyb1.ELL_logical_width != myhyb2.ELL_logical_width) {
    return false;
  }
  // ELL physical width could be different
  for (int64_t IdxNode = 0; IdxNode < myhyb1.HybIndexMax; IdxNode++) {
    std::set<std::pair<IdxType, std::pair<IdxType, IdxType>>>
        DestNodeRelTypeEidsSet;
    for (int64_t IdxElement = IdxNode * myhyb1.ELL_physical_width;
         IdxElement <
         IdxNode * myhyb1.ELL_physical_width + myhyb1.ELL_logical_width;
         IdxElement++) {
      IdxType IdxDestNode = myhyb1.ELLColIdx[IdxElement];
      IdxType IdxRelationship = myhyb1.ELLRelType[IdxElement];
      IdxType IdxEids = myhyb1.ELLEids[IdxElement];
      if (IdxDestNode == MyHyb_NONEXISTENT_ELEMENT) {
        continue;
      }
      DestNodeRelTypeEidsSet.insert(std::make_pair(
          IdxDestNode, std::make_pair(IdxRelationship, IdxEids)));
    }
    int64_t NumEdgesFromThisSourceNode1 = DestNodeRelTypeEidsSet.size();
    for (int64_t IdxElement = IdxNode * myhyb2.ELL_physical_width;
         IdxElement <
         IdxNode * myhyb2.ELL_physical_width + myhyb2.ELL_logical_width;
         IdxElement++) {
      IdxType IdxDestNode = myhyb2.ELLColIdx[IdxElement];
      IdxType IdxRelationship = myhyb2.ELLRelType[IdxElement];
      IdxType IdxEids = myhyb2.ELLEids[IdxElement];
      if (IdxDestNode == MyHyb_NONEXISTENT_ELEMENT) {
        continue;
      }
      DestNodeRelTypeEidsSet.insert(std::make_pair(
          IdxDestNode, std::make_pair(IdxRelationship, IdxEids)));
    }
    // check if the number of edges from this source node is the same
    if (DestNodeRelTypeEidsSet.size() != NumEdgesFromThisSourceNode1) {
      return false;
    }
  }

  return IsEqual(myhyb1.csr, myhyb2.csr);
}

template <typename IdxType>
MyHyb<IdxType, std::allocator<IdxType>,
      MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>>
IntegratedCSRToHyb_CPU(
    const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> &csr,
    int64_t ELL_logical_width, int64_t ELL_physical_width,
    int64_t ELLMaxIndex) {
  // TODO: implement here
  thrust::host_vector<IdxType> ELLColIdx(ELLMaxIndex * ELL_physical_width,
                                         MyHyb_NONEXISTENT_ELEMENT);
  thrust::host_vector<IdxType> ELLRelType(ELLMaxIndex * ELL_physical_width,
                                          MyHyb_NONEXISTENT_ELEMENT);
  thrust::host_vector<IdxType> ELLEids(ELLMaxIndex * ELL_physical_width,
                                       MyHyb_NONEXISTENT_ELEMENT);

  // based on MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>
  // ToSeparateCSR_CPU(const MyHeteroIntegratedCSR<IdxType,
  // std::allocator<IdxType>>& csr)

  thrust::host_vector<IdxType> result_row_ptrs(csr.num_rows * csr.num_rels + 1,
                                               0);
  thrust::host_vector<IdxType> result_col_indices(csr.total_num_nnzs, 0);
  thrust::host_vector<IdxType> result_csr_eids(csr.total_num_nnzs, 0);
  // TODO: implement here

  std::vector<std::vector<std::vector<std::pair<IdxType, IdxType>>>>
      residue_csr_rel_type_to_edges(
          csr.num_rels,
          std::vector<std::vector<std::pair<IdxType, IdxType>>>(csr.num_rows));

  for (int64_t IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
    for (IdxType IdxEdge = csr.row_ptrs[IdxRow];
         IdxEdge < csr.row_ptrs[IdxRow + 1]; IdxEdge++) {
      IdxType IdxSrcNode = IdxRow;
      IdxType IdxDestNode = csr.col_indices[IdxEdge];
      IdxType IdxRelationshipEdge = csr.rel_type[IdxEdge];
      IdxType IdxEids = csr.eids[IdxEdge];
      if (IdxEdge - csr.row_ptrs[IdxRow] < ELL_logical_width &&
          IdxRow < ELLMaxIndex) {
        // store the edge in the ELL
        ELLColIdx[IdxRow * ELL_physical_width + IdxEdge -
                  csr.row_ptrs[IdxRow]] = IdxDestNode;
        ELLRelType[IdxRow * ELL_physical_width + IdxEdge -
                   csr.row_ptrs[IdxRow]] = IdxRelationshipEdge;
        ELLEids[IdxRow * ELL_physical_width + IdxEdge - csr.row_ptrs[IdxRow]] =
            IdxEids;
      } else {
        // store the rest into the CSR
        residue_csr_rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode]
            .push_back(std::make_pair(IdxDestNode, IdxEids));
      }
    }
  }

  int64_t csr_total_num_nnz = 0;
  for (int64_t IdxRelationship = 0; IdxRelationship < csr.num_rels;
       IdxRelationship++) {
    for (IdxType IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
      result_row_ptrs[IdxRelationship * csr.num_rows + IdxRow + 1] =
          result_row_ptrs[IdxRelationship * csr.num_rows + IdxRow];
      for (IdxType IdxElement = 0;
           IdxElement <
           residue_csr_rel_type_to_edges[IdxRelationship][IdxRow].size();
           IdxElement++) {
        result_col_indices[result_row_ptrs[IdxRelationship * csr.num_rows +
                                           IdxRow + 1]] =
            residue_csr_rel_type_to_edges[IdxRelationship][IdxRow][IdxElement]
                .first;
        result_csr_eids[result_row_ptrs[IdxRelationship * csr.num_rows +
                                        IdxRow + 1]] =
            residue_csr_rel_type_to_edges[IdxRelationship][IdxRow][IdxElement]
                .second;
        result_row_ptrs[IdxRelationship * csr.num_rows + IdxRow + 1] += 1;
        csr_total_num_nnz += 1;
      }
    }
  }

  // resize csr vectors
  result_csr_eids.resize(csr_total_num_nnz);
  result_col_indices.resize(csr_total_num_nnz);

  MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> resultCSR(
      csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptrs,
      result_col_indices, result_csr_eids);
  MyHyb<IdxType, std::allocator<IdxType>,
        MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>>
      result_hyb(ELLMaxIndex, ELL_logical_width, ELL_physical_width, resultCSR,
                 ELLColIdx, ELLRelType, ELLEids, csr.num_rows, csr.num_cols,
                 csr.num_rels, csr.num_nnzs);
  return result_hyb;
}

template <typename IdxType>
MyHyb<IdxType, std::allocator<IdxType>,
      MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>>
IntegratedCSRToHyb_ADHOC_CPU(
    const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> &csr,
    int64_t ELL_logical_width, int64_t ELL_physical_width,
    int64_t ELLMaxIndex) {
  // TODO: this is an ad hoc solution
  // TODO: implement here
  thrust::host_vector<IdxType> ELLColIdx(ELLMaxIndex * ELL_physical_width,
                                         MyHyb_NONEXISTENT_ELEMENT);
  thrust::host_vector<IdxType> ELLRelType(ELLMaxIndex * ELL_physical_width,
                                          MyHyb_NONEXISTENT_ELEMENT);
  thrust::host_vector<IdxType> ELLEids(ELLMaxIndex * ELL_physical_width,
                                       MyHyb_NONEXISTENT_ELEMENT);
  // based on MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>
  // ToSeparateCSR_CPU(const MyHeteroIntegratedCSR<IdxType,
  // std::allocator<IdxType>>& csr)

  thrust::host_vector<IdxType> result_row_ptrs(csr.num_rows + 1, 0);
  thrust::host_vector<IdxType> result_col_indices(csr.total_num_nnzs, 0);
  thrust::host_vector<IdxType> result_rel_type(csr.total_num_nnzs, 0);
  thrust::host_vector<IdxType> result_eids(csr.total_num_nnzs, 0);
  // TODO: implement here
  printf("csr_total_num_nnzs: %ld\n", csr.total_num_nnzs);
  printf("csr.num_rows: %ld\n", csr.num_rows);
  thrust::host_vector<
      thrust::host_vector<std::pair<IdxType, std::pair<IdxType, IdxType>>>>
      residue_csr_src_nodes_to_dests_with_rel_type_and_eids;
  residue_csr_src_nodes_to_dests_with_rel_type_and_eids.resize(csr.num_rows);
  for (int64_t IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
    for (IdxType IdxEdge = csr.row_ptrs[IdxRow];
         IdxEdge < csr.row_ptrs[IdxRow + 1]; IdxEdge++) {
      IdxType IdxSrcNode = IdxRow;
      IdxType IdxDestNode = csr.col_indices[IdxEdge];
      IdxType IdxRelationshipEdge = csr.rel_type[IdxEdge];
      IdxType IdxEid = csr.eids[IdxEdge];
      if (IdxEdge - csr.row_ptrs[IdxRow] < ELL_logical_width &&
          IdxRow < ELLMaxIndex) {
        // store the edge in the ELL
        ELLColIdx[IdxRow * ELL_physical_width + IdxEdge -
                  csr.row_ptrs[IdxRow]] = IdxDestNode;
        ELLRelType[IdxRow * ELL_physical_width + IdxEdge -
                   csr.row_ptrs[IdxRow]] = IdxRelationshipEdge;
        ELLEids[IdxRow * ELL_physical_width + IdxEdge - csr.row_ptrs[IdxRow]] =
            IdxEid;
      } else {
        // store the rest into the CSR
        residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxSrcNode]
            .push_back(std::make_pair(
                IdxDestNode, std::make_pair(IdxRelationshipEdge, IdxEid)));
      }
    }
  }

  int64_t csr_total_num_nnz = 0;

  for (IdxType IdxRow = 0; IdxRow < csr.num_rows; IdxRow++) {
    result_row_ptrs[IdxRow + 1] = result_row_ptrs[IdxRow];
    for (IdxType IdxElement = 0;
         IdxElement <
         residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxRow].size();
         IdxElement++) {
      result_col_indices[result_row_ptrs[IdxRow + 1]] =
          residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxRow]
                                                               [IdxElement]
                                                                   .first;
      result_rel_type[result_row_ptrs[IdxRow + 1]] =
          residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxRow]
                                                               [IdxElement]
                                                                   .second
                                                                   .first;
      result_eids[result_row_ptrs[IdxRow + 1]] =
          residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxRow]
                                                               [IdxElement]
                                                                   .second
                                                                   .second;
      result_row_ptrs[IdxRow + 1] += 1;
      csr_total_num_nnz += 1;
    }
  }

  // resize
  result_eids.resize(csr_total_num_nnz);
  result_col_indices.resize(csr_total_num_nnz);
  result_rel_type.resize(csr_total_num_nnz);

  MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> resultCSR(
      csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptrs,
      result_col_indices, result_rel_type, result_eids);
  MyHyb<IdxType, std::allocator<IdxType>,
        MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>>
      result_hyb(ELLMaxIndex, ELL_logical_width, ELL_physical_width, resultCSR,
                 ELLColIdx, ELLRelType, ELLEids, csr.num_rows, csr.num_cols,
                 csr.num_rels, csr.num_nnzs);
  return result_hyb;
}

template <typename IdxType>
// MyHyb<IdxType, std::allocator<IdxType>, MyHeteroIntegratedCSR<IdxType,
// thrust::device_allocator<IdxType>>>
MyHyb<IdxType, thrust::device_allocator<IdxType>,
      MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>>
IntegratedCSRToHyb_GPU(
    const MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>>
        &csr,
    int64_t ELL_logical_width, int64_t ELL_physical_width,
    int64_t ELLMaxIndex) {
  thrust::device_vector<IdxType> ELLColIdx(ELLMaxIndex * ELL_physical_width,
                                           MyHyb_NONEXISTENT_ELEMENT);
  thrust::device_vector<IdxType> ELLRelType(ELLMaxIndex * ELL_physical_width,
                                            MyHyb_NONEXISTENT_ELEMENT);
  // based on MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>
  // ToSeparateCSR_CPU(const MyHeteroIntegratedCSR<IdxType,
  // std::allocator<IdxType>>& csr)

  thrust::device_vector<IdxType> result_row_ptrs(
      csr.num_rows * csr.num_rels + 1, 0);
  thrust::device_vector<IdxType> result_col_indices(csr.num_nnzs, 0);

  // TODO: implement here
  assert(0 && "GPU kernel not implemented");
  // return dummy to avoid compiler error
  MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>> resultCSR(
      csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptrs,
      result_col_indices);
  MyHyb<IdxType, thrust::device_allocator<IdxType>,
        MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>>
      result_hyb(ELLMaxIndex, ELL_logical_width, ELL_physical_width, resultCSR,
                 csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs);
  return result_hyb;
}

template <typename IdxType>
void SeparateCSRToHyb_CPU(
    const MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> &csr) {
  // TODO: implement here
  assert(0 && "SeparateCSR to Hyb kernel not implemented");
}

template <typename IdxType>
// MyHyb<IdxType, std::allocator<IdxType>, MyHeteroSeparateCSR<IdxType,
// thrust::device_allocator<IdxType>>>
void SeparateCSRToHyb_GPU(
    const MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>
        &csr) {
  // TODO: implement here
  assert(0 && "SeparateCSR to Hyb kernel not implemented");
  // return dummy to avoid compiler error
}

template <typename InputAlloc, typename ReturnAlloc, typename IdxType>
MyHeteroIntegratedCSR<IdxType, ReturnAlloc> ToIntegratedCSR(
    const MyHeteroSeparateCSR<IdxType, InputAlloc> &csr) {
  // get cpu and gpu implementation.
  if (csr.IsDataOnCPU()) {
    auto ret_value = ToIntegratedCSR_CPU<IdxType>(csr);
    return ret_value;  // implicit type conversion here to match return
                       // ReturnAlloc type.
  } else {
    assert(csr.IsDataOnGPU());
    auto ret_value = ToIntegratedCSR_GPU<IdxType>(csr);
    return ret_value;  // implicit type conversion here to match return
                       // ReturnAlloc type.
  }
}

template <typename InputAlloc, typename ReturnAlloc, typename IdxType>
MyHeteroSeparateCSR<IdxType, ReturnAlloc> ToSeparateCSR(
    const MyHeteroIntegratedCSR<IdxType, InputAlloc> &csr) {
  // get cpu and gpu implementation.
  if (csr.IsDataOnCPU()) {
    auto ret_value = ToSeparateCSR_CPU<IdxType>(csr);
    return ret_value;  // implicit type conversion here to match return
                       // ReturnAlloc type.
  } else {
    assert(csr.IsDataOnGPU());
    auto ret_value = ToSeparateCSR_GPU<IdxType>(csr);
    return ret_value;  // implicit type conversion here to match return
                       // ReturnAlloc type.
  }
}

template <typename InputAlloc, typename ReturnAlloc, typename IdxType>
MyHyb<IdxType, ReturnAlloc, MyHeteroSeparateCSR<IdxType, ReturnAlloc>>
MyHybIntegratedCSRToSeparateCSR(
    const MyHyb<IdxType, ReturnAlloc,
                MyHeteroIntegratedCSR<IdxType, ReturnAlloc>> &hyb) {
  MyHyb<IdxType, ReturnAlloc, MyHeteroSeparateCSR<IdxType, ReturnAlloc>> result(
      hyb.HybIndexMax, hyb.ELL_logical_width, hyb.ELL_physical_width,
      ToSeparateCSR<InputAlloc, ReturnAlloc, IdxType>(hyb.csr), hyb.ELLColIdx,
      hyb.ELLRelType, hyb.num_rows, hyb.num_cols, hyb.num_rels, hyb.num_nnzs);
  return result;
}