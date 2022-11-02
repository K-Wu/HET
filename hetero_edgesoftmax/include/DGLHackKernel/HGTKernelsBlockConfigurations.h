#pragma once
#include "DGLHackKernel.h"

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
            num_blocks_along_dimx_for_all_prev_relation_vect.size() - 1 &&
        idx_block >= num_blocks_along_dimx_for_all_prev_relation_vect
                         [idx_curr_relation]) {
      assert(curr_beg_node_entry_idx / num_node_per_block_per_iteration ==
             num_blocks_along_dimx_for_same_relation_vect[idx_curr_relation]);
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

template <bool EqualPartitionFlag, typename IteratorType>
std::pair<std::vector<int>, std::vector<int>>
get_schedule_by_relation_kernel_launch_metadata(
    int num_blocks_along_dimx, int num_relations,
    IteratorType num_job_entries_per_relation_beg,
    IteratorType num_job_entries_per_relation_end) {
  std::vector<int> num_blocks_along_dimx_for_same_relation_vect;
  std::vector<int> num_blocks_along_dimx_for_all_prev_relation_vect;
  num_blocks_along_dimx_for_all_prev_relation_vect.push_back(0);

  if (EqualPartitionFlag) {
    // for ease of programming equally partition the workload to different
    // blocks at this moment.
    for (int idx_relationship = 0; idx_relationship < num_relations;
         idx_relationship++) {
      int num_blocks_along_dimx_for_this_and_prev_relation =
          (idx_relationship + 1 + 0.0) / (num_relations + 0.0) *
          num_blocks_along_dimx;
      num_blocks_along_dimx_for_all_prev_relation_vect.push_back(
          num_blocks_along_dimx_for_this_and_prev_relation);
    }

  } else {
    int total_num_job_entries = 0;
    for (IteratorType iter = num_job_entries_per_relation_beg;
         iter != num_job_entries_per_relation_end; iter++) {
      total_num_job_entries += *iter;
    }
    int num_job_entries_for_this_and_prev_relation = 0;
    IteratorType curr_iter = num_job_entries_per_relation_beg;
    // in this branch, let's allocate blocks according to the amount of workload
    for (int idx_relationship = 0; idx_relationship < num_relations;
         idx_relationship++) {
      int num_job_entries_for_current_relation = *curr_iter;
      num_job_entries_for_this_and_prev_relation +=
          num_job_entries_for_current_relation;

      int num_blocks_along_dimx_for_this_and_prev_relation =
          (num_job_entries_for_this_and_prev_relation + 0.0) /
          (total_num_job_entries)*num_blocks_along_dimx;
      if (num_blocks_along_dimx_for_this_and_prev_relation ==
          num_blocks_along_dimx_for_all_prev_relation_vect
              [num_blocks_along_dimx_for_all_prev_relation_vect.size() - 1]) {
        // if there is too few jobs for current relation, we still need to
        // assign at least one block to it.
        num_blocks_along_dimx_for_this_and_prev_relation += 1;
      }
      num_blocks_along_dimx_for_all_prev_relation_vect.push_back(
          num_blocks_along_dimx_for_this_and_prev_relation);
      num_blocks_along_dimx_for_same_relation_vect.push_back(
          num_blocks_along_dimx_for_this_and_prev_relation -
          num_blocks_along_dimx_for_all_prev_relation_vect
              [num_blocks_along_dimx_for_all_prev_relation_vect.size() - 2]);

      curr_iter++;
    }
    if (num_blocks_along_dimx_for_all_prev_relation_vect
            [num_blocks_along_dimx_for_all_prev_relation_vect.size() - 1] !=
        num_blocks_along_dimx) {
      printf(
          "WARNING: we have corrected the number of blocks from %d to %d in "
          "order to make sure each relation get at least 1 blocks in "
          "get_schedule_by_relation_kernel_launch_metadata()",
          num_blocks_along_dimx,
          num_blocks_along_dimx_for_all_prev_relation_vect
              [num_blocks_along_dimx_for_all_prev_relation_vect.size() - 1]);
    }
  }

  for (int idx_relationship = 0; idx_relationship < num_relations;
       idx_relationship++) {
    num_blocks_along_dimx_for_same_relation_vect.push_back(
        num_blocks_along_dimx_for_all_prev_relation_vect[idx_relationship + 1] -
        num_blocks_along_dimx_for_all_prev_relation_vect[idx_relationship]);
  }
  num_blocks_along_dimx_for_all_prev_relation_vect.erase(
      num_blocks_along_dimx_for_all_prev_relation_vect.begin());

  return std::make_pair(num_blocks_along_dimx_for_same_relation_vect,
                        num_blocks_along_dimx_for_all_prev_relation_vect);
}
