#pragma once
#include "../DGLHackKernel.h"
// This file provides logic to generate assignments from relationship to work,
// for gemm when it is applied to node entries or edge entries. The logic can be
// used for my_shmem_sgemm.cu.h and mysgemm_functor.cu.h

std::tuple<thrust::device_vector<int>, thrust::device_vector<int>,
           thrust::device_vector<int>>
get_schedule_by_relation_kernel_launch_per_block_metadata(
    std::vector<int>& num_blocks_along_dimx_for_same_relation_vect,
    std::vector<int>& num_blocks_along_dimx_for_all_prev_relation_vect,
    int num_blocks_along_dimx, int num_node_per_block_per_iteration) {
  thrust::device_vector<int>
      num_blocks_along_dimx_for_same_relation_per_block_vect;
  thrust::device_vector<int> blockid_relation_id_vect;
  thrust::device_vector<int> beg_node_entry_idxes_vect;

  int idx_curr_relation = 0;
  int curr_beg_node_entry_idx = 0;
  for (int idx_block = 0; idx_block < num_blocks_along_dimx; idx_block++) {
    if (idx_curr_relation <
            num_blocks_along_dimx_for_all_prev_relation_vect.size() - 2 &&
        idx_block >=
            num_blocks_along_dimx_for_all_prev_relation_vect[idx_curr_relation +
                                                             1]) {
      idx_curr_relation++;
      // NB: current implementation assumes the node entry index starts from 0
      // whenever the iteration of dealing with a new relation begins. May need
      // to generalize this to edge entry and cases where the index of a new
      // relation starts with an offset.
      curr_beg_node_entry_idx = 0;
    }
    blockid_relation_id_vect.push_back(idx_curr_relation);
    beg_node_entry_idxes_vect.push_back(curr_beg_node_entry_idx);
    curr_beg_node_entry_idx += num_node_per_block_per_iteration;
    num_blocks_along_dimx_for_same_relation_per_block_vect.push_back(
        num_blocks_along_dimx_for_same_relation_vect[idx_curr_relation]);
  }

  return std::make_tuple(num_blocks_along_dimx_for_same_relation_per_block_vect,
                         blockid_relation_id_vect, beg_node_entry_idxes_vect);
}

template <bool EqualPartitionFlag, bool PartitionAccordingToBlockSizeFlag,
          typename IteratorType>
std::pair<std::vector<int>, std::vector<int>>
get_schedule_by_relation_kernel_launch_metadata(
    int num_blocks_along_dimx, int num_relations, int block_size,
    IteratorType num_job_entries_for_all_prev_relation_beg,
    IteratorType num_job_entries_for_all_prev_relation_end) {
  // IteratorType num_job_entries_per_relation_beg,
  // IteratorType num_job_entries_per_relation_end) {
  std::vector<int> num_blocks_along_dimx_for_same_relation_vect;
  std::vector<int> num_blocks_along_dimx_for_all_prev_relation_vect;
  num_blocks_along_dimx_for_all_prev_relation_vect.push_back(0);

  if constexpr (EqualPartitionFlag) {
    // for ease of programming equally partition the workload to different
    // blocks at this moment.
    assert(num_blocks_along_dimx > 0);
    for (int idx_relationship = 0; idx_relationship < num_relations;
         idx_relationship++) {
      int num_blocks_along_dimx_for_this_and_prev_relation =
          (idx_relationship + 1 + 0.0) / (num_relations + 0.0) *
          num_blocks_along_dimx;
      num_blocks_along_dimx_for_all_prev_relation_vect.push_back(
          num_blocks_along_dimx_for_this_and_prev_relation);
    }

  } else {
    if constexpr (PartitionAccordingToBlockSizeFlag) {
      IteratorType num_job_entries_for_all_prev_iter =
          num_job_entries_for_all_prev_relation_beg + 1;
      int num_job_entries_for_all_prev = 0;
      for (int idx_relationship = 0; idx_relationship < num_relations;
           idx_relationship++) {
        int num_job_entries =
            (*num_job_entries_for_all_prev_iter) - num_job_entries_for_all_prev;
        num_job_entries_for_all_prev = *num_job_entries_for_all_prev_iter;
        num_job_entries_for_all_prev_iter++;
        int num_blocks_along_dimx_for_this_relation =
            (num_job_entries + block_size - 1) / block_size;
        num_blocks_along_dimx_for_same_relation_vect.push_back(
            num_blocks_along_dimx_for_this_relation +
            num_blocks_along_dimx_for_all_prev_relation_vect.back());
      }
    } else {
      // in this branch, let's allocate blocks according to the amount of
      // workload and the number of blocks.
      assert(num_blocks_along_dimx > 0);

      int total_num_job_entries =
          *(num_job_entries_for_all_prev_relation_beg + num_relations);

      int num_job_entries_for_this_and_prev_relation = 0;

      IteratorType curr_iter = num_job_entries_for_all_prev_relation_beg + 1;

      for (int idx_relationship = 0; idx_relationship < num_relations;
           idx_relationship++) {
        int num_job_entries_for_current_relation =
            *curr_iter - num_job_entries_for_this_and_prev_relation;
        num_job_entries_for_this_and_prev_relation = *curr_iter;

        int num_blocks_along_dimx_for_this_and_prev_relation =
            (num_job_entries_for_this_and_prev_relation + 0.0) /
            (total_num_job_entries)*num_blocks_along_dimx;
        if (num_blocks_along_dimx_for_this_and_prev_relation <=
            num_blocks_along_dimx_for_all_prev_relation_vect.back()) {
          // if there is too few jobs for current relation, we still need to
          // assign at least one block to it.
          num_blocks_along_dimx_for_this_and_prev_relation =
              num_blocks_along_dimx_for_all_prev_relation_vect.back() + 1;
        }
        num_blocks_along_dimx_for_all_prev_relation_vect.push_back(
            num_blocks_along_dimx_for_this_and_prev_relation);
        num_blocks_along_dimx_for_same_relation_vect.push_back(
            num_blocks_along_dimx_for_this_and_prev_relation -
            num_blocks_along_dimx_for_all_prev_relation_vect.back());

        curr_iter++;
      }
    }
    if (num_blocks_along_dimx_for_all_prev_relation_vect.back() !=
        num_blocks_along_dimx) {
      printf(
          "WARNING: we have corrected the number of blocks from %d to %d in "
          "order to make sure each relation (%d) get at least 1 blocks in "
          "get_schedule_by_relation_kernel_launch_metadata()",
          num_blocks_along_dimx,
          num_blocks_along_dimx_for_all_prev_relation_vect.back(),
          num_relations);
    }
  }

  for (int idx_relationship = 0; idx_relationship < num_relations;
       idx_relationship++) {
    num_blocks_along_dimx_for_same_relation_vect.push_back(
        num_blocks_along_dimx_for_all_prev_relation_vect[idx_relationship + 1] -
        num_blocks_along_dimx_for_all_prev_relation_vect[idx_relationship]);
  }
  // num_blocks_along_dimx_for_all_prev_relation_vect.erase(
  //    num_blocks_along_dimx_for_all_prev_relation_vect.begin());

  return std::make_pair(num_blocks_along_dimx_for_same_relation_vect,
                        num_blocks_along_dimx_for_all_prev_relation_vect);
}
