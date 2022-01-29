#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <npy.hpp>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <chrono>
#include <curand.h>

#define RTX_3090_BLOCKSIZE 1024
#define RTX_3090_SM_NUM 82

__device__ void _perRow_EdgeSoftmaxFirstStageCSRKernel(int row_idx, float * __restrict__ outNode,  int num_rows, int nnz, int * __restrict__ matCols, int * __restrict__ matRows,
                              float * __restrict__ edge_input_data, float mu) {
  //@@ insert spmv kernel for csr format 
    if (row_idx >= num_rows) return;
    int row_start = matRows[row_idx];
    int row_end = matRows[row_idx + 1];
    //float row_sum = 1e-10f;
    for (int i = row_start; i < row_end; i++) {
        int col_idx = matCols[i];
        float val = expf(edge_input_data[i])+1e-10f;
        outNode[col_idx] += val;
    }
}

__device__ void _perRow_EdgeSoftmaxSecondStageCSRKernel(int row_idx, float* __restrict__ outEdge, float* __restrict__ outNode, int num_rows, int nnz, int* __restrict__ matCols, int* __restrict__ matRows,
    float* __restrict__ edge_input_data, float mu) {
    //@@ insert spmv kernel for csr format 
    if (row_idx >= num_rows) return;
    int row_start = matRows[row_idx];
    int row_end = matRows[row_idx + 1];

    for (int i = row_start; i<row_end; i++) {
        int col_idx = matCols[i];
        float val = mu*expf(edge_input_data[i]) / outNode[col_idx];
        outEdge[i] = val;
    }
}

__device__ void _EdgeSoftmaxFirstStageCSRKernel(int beg_row_idx, int stride, float* __restrict__ outNode, int num_rows, int nnz, int * __restrict__ matCols, int * __restrict__ matRows,
                              float * __restrict__ edge_input_data, float mu){
  for (int row_idx = beg_row_idx; row_idx < num_rows; row_idx += stride) {
    _perRow_EdgeSoftmaxFirstStageCSRKernel(row_idx, outNode, num_rows, nnz, matCols, matRows, edge_input_data, mu);
  }
}

__device__ void _EdgeSoftmaxSecondStageCSRKernel(int beg_row_idx, int stride, float* __restrict__ outEdge, float* __restrict__ outNode, int num_rows, int nnz, int* __restrict__ matCols, int* __restrict__ matRows,
    float* __restrict__ edge_input_data, float mu) {
    for (int row_idx = beg_row_idx; row_idx < num_rows; row_idx += stride) {
        _perRow_EdgeSoftmaxSecondStageCSRKernel(row_idx, outEdge, outNode, num_rows, nnz, matCols, matRows, edge_input_data, mu);
    }
}

__device__ void _perRow_EdgeSoftmaxFirstStageConcatenatedCSRKernel(int row_idx, float** __restrict__ outNodes_per_relation, int num_rows, int nnz, int* __restrict__ matCols, int* __restrict__ matRows, int* __restrict__ relation,
    float* __restrict__ edge_input_data, float* mus) {
    //@@ insert spmv kernel for csr format 
    if (row_idx >= num_rows) return;
    int row_start = matRows[row_idx];
    int row_end = matRows[row_idx + 1];
    //float row_sum = 1e-10f;
    for (int i = row_start; i < row_end; i++) {
        int col_idx = matCols[i];
        float val = expf(edge_input_data[i]) + 1e-10f;
        outNodes_per_relation[relation[col_idx]][col_idx] += val;
    }
}

__device__ void _perRow_EdgeSoftmaxSecondStageConcatenatedCSRKernel(int row_idx, float* __restrict__ outEdge, float** __restrict__ outNodes_per_relation, int num_rows, int nnz, int* __restrict__ matCols, int* __restrict__ matRows,
    int* __restrict__ relation, float* __restrict__ edge_input_data, float* mus) {
    //@@ insert spmv kernel for csr format 
    if (row_idx >= num_rows) return;
    int row_start = matRows[row_idx];
    int row_end = matRows[row_idx + 1];

    for (int i = row_start; i < row_end; i++) {
        int col_idx = matCols[i];
        float val = mus[relation[col_idx]] * expf(edge_input_data[i]) / outNodes_per_relation[relation[col_idx]][col_idx];
        outEdge[i] = val;
    }
}

__device__ void _EdgeSoftmaxFirstStageConcatenatedCSRKernel(int beg_row_idx, int stride, float** __restrict__ outNodes_per_relation, int num_rows, int nnz, int* __restrict__ matCols, int* __restrict__ matRows, int* __restrict__ matRelation,
    float* __restrict__ edge_input_data, float* mus) {
    for (int row_idx = beg_row_idx; row_idx < num_rows; row_idx += stride) {
        _perRow_EdgeSoftmaxFirstStageConcatenatedCSRKernel(row_idx, outNodes_per_relation, num_rows, nnz, matCols, matRows, matRelation, edge_input_data, mus);
    }
}

__device__ void _EdgeSoftmaxSecondStageConcatenatedCSRKernel(int beg_row_idx, int stride, float* __restrict__ outEdge, float** __restrict__ outNodes_per_relation, int num_rows, int nnz, int* __restrict__ matCols, int* __restrict__ matRows, int* __restrict__ matRelation,
    float* __restrict__ edge_input_data, float* mus) {
    for (int row_idx = beg_row_idx; row_idx < num_rows; row_idx += stride) {
        _perRow_EdgeSoftmaxSecondStageConcatenatedCSRKernel(row_idx, outEdge, outNodes_per_relation, num_rows, nnz, matCols, matRows, matRelation, edge_input_data, mus);
    }
}

__global__ void EdgeSoftmaxConcatenatedCSRKernel(float * __restrict__ outEdges, float** __restrict__ outNodes_per_relation, int num_rows, int nnz, int * __restrict__ matCols, int * __restrict__ matRows, int* __restrict__ matRelation,
                              float * __restrict__ edge_input_data, float* mus) {
  int beg_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  _EdgeSoftmaxFirstStageConcatenatedCSRKernel(beg_row_idx, blockDim.x * gridDim.x, outNodes_per_relation, num_rows, nnz, matCols, matRows, matRelation, edge_input_data, mus);
  _EdgeSoftmaxSecondStageConcatenatedCSRKernel(beg_row_idx, blockDim.x * gridDim.x, outEdges, outNodes_per_relation, num_rows, nnz, matCols, matRows, matRelation, edge_input_data, mus);
}

__global__ void EdgeSoftmaxMultiRelationsCSRSKernel(int* smid_relation_id, int* beg_row_idxes, int* num_smid_for_same_relation, float **outNode_per_relation, float* outEdge, int num_relations, int num_rows, int* nnzs, int ** matCols, int ** matRows,
                              float *edge_input_data, float* mus) {
   int smid = getsmid();
   int relation_id = smid_relation_id[smid];
   int offset = 0;
   for (int idx = 0; idx<relation_id; idx++) {
       offset += nnzs[idx];
   }
   _EdgeSoftmaxFirstStageCSRKernel(beg_row_idxes[smid],num_smid_for_same_relation[smid]*blockDim.x,outNode_per_relation[relation_id], num_rows, nnzs[relation_id], matCols[relation_id], matRows[relation_id], &edge_input_data[offset], mus[relation_id]);
   _EdgeSoftmaxSecondStageCSRKernel(beg_row_idxes[smid], num_smid_for_same_relation[smid] * blockDim.x, &outEdge[offset], outNode_per_relation[relation_id], num_rows, nnzs[relation_id], matCols[relation_id], matRows[relation_id], &edge_input_data[offset], mus[relation_id]);
}

void doGPUEdgeSoftmaxConcatenatedCSRKernel(cusp::csr_matrix<int, int, cusp::device_memory> concatenated_csr_matrix, int num_relations) {

    std::vector<thrust::device_vector<float>> outNodes_per_relation_vect_vect;
    thrust::device_vector<float*> outNodes_per_relation_vect;

    for (int idx_matrix = 0; idx_matrix < num_relations; idx_matrix++) {
        thrust::device_vector<float> outEdge_vect_for_curr_relation(concatenated_csr_matrix.num_rows, 0);
        outNodes_per_relation_vect_vect.push_back(outEdge_vect_for_curr_relation);
        outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(outEdge_vect_for_curr_relation.data()));
    }

    float* outEdges;
    float* mus;
    float* edge_input_data;
    cudaMalloc((void**)&edge_input_data, sizeof(float) * concatenated_csr_matrix.column_indices.size());
    cudaMalloc((void**)&mus, sizeof(float) * num_relations);
    cudaMalloc((void**)&outEdges, sizeof(float) * concatenated_csr_matrix.column_indices.size());

    curandGenerator_t m_prng;
    //Create a new generator
    curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
    //Set the generator options
    curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long)0);
    //Generate random numbers
    curandGenerateUniform(m_prng, mus, num_relations);
    curandGenerateUniform(m_prng, edge_input_data, concatenated_csr_matrix.column_indices.size());






    dim3 block(RTX_3090_BLOCKSIZE, 1, 1);
    dim3 grid(RTX_3090_SM_NUM, 1, 1);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    EdgeSoftmaxConcatenatedCSRKernel << <grid, block >> > ( outEdges, thrust::raw_pointer_cast(outNodes_per_relation_vect.data()),   concatenated_csr_matrix.num_rows,  concatenated_csr_matrix.column_indices.size(), thrust::raw_pointer_cast(concatenated_csr_matrix.column_indices.data()), thrust::raw_pointer_cast(concatenated_csr_matrix.row_offsets.data()), thrust::raw_pointer_cast(concatenated_csr_matrix.values.data()), edge_input_data, mus);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU doGPUEdgeSoftmaxConcatenatedCSRKernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    cudaFree(outEdges);
    cudaFree(edge_input_data);
    cudaFree(mus);

}

void doGPUEdgeSoftmaxMultiRelationsCSRSKernel(std::vector<cusp::csr_matrix<int, int, cusp::device_memory>> csr_matrices) {
    thrust::device_vector<int*> matCols_vect = thrust::device_vector<int*>(csr_matrices.size());
    thrust::device_vector<int*> matRows_vect = thrust::device_vector<int*>(csr_matrices.size());
    thrust::device_vector<int> nnzs_vect;
    thrust::device_vector<int> smid_relation_id_vect;
    thrust::device_vector<int> beg_row_idxes_vect;
        thrust::device_vector<int> num_sms_for_same_relation_per_sm_vect;
        std::vector<thrust::device_vector<float>> outNodes_per_relation_vect_vect;
        thrust::device_vector<float*> outNodes_per_relation_vect;
    size_t total_nnzs = 0;
    for (int idx_matrix = 0; idx_matrix < csr_matrices.size(); idx_matrix++) {
        
        matCols_vect[idx_matrix] = thrust::raw_pointer_cast(csr_matrices[idx_matrix].column_indices.data());
        matRows_vect[idx_matrix] = thrust::raw_pointer_cast(csr_matrices[idx_matrix].row_offsets.data());
        nnzs_vect.push_back(csr_matrices[idx_matrix].column_indices.size());
        total_nnzs+=csr_matrices[idx_matrix].column_indices.size();
    }

    for (int idx_matrix = 0; idx_matrix < csr_matrices.size(); idx_matrix++) {
        thrust::device_vector<float> outEdge_vect_for_curr_relation( csr_matrices[0].num_rows, 0);
        outNodes_per_relation_vect_vect.push_back(outEdge_vect_for_curr_relation);
        outNodes_per_relation_vect.push_back(thrust::raw_pointer_cast(outEdge_vect_for_curr_relation.data()));
    }

    float* outEdges;
    float* mus;
    float* edge_input_data;
    cudaMalloc((void**)&edge_input_data,sizeof(float)*total_nnzs);
    cudaMalloc((void**)&mus,sizeof(float)*csr_matrices.size());
    cudaMalloc((void**)&outEdges,sizeof(float)*total_nnzs);

    curandGenerator_t m_prng;
    //Create a new generator
    curandCreateGenerator(&m_prng, CURAND_RNG_PSEUDO_DEFAULT);
    //Set the generator options
    curandSetPseudoRandomGeneratorSeed(m_prng, (unsigned long) 0);
    //Generate random numbers
    curandGenerateUniform(m_prng, mus, csr_matrices.size());
    curandGenerateUniform(m_prng, edge_input_data, total_nnzs);
    



    std::vector<int> num_sms_for_same_relation_vect;
    std::vector<int> num_sms_for_all_prev_relation_vect;
    num_sms_for_all_prev_relation_vect.push_back(0);
    for (int idx_relationship = 0; idx_relationship < csr_matrices.size(); idx_relationship++) {
        int num_sms_for_this_and_prev_relation= (idx_relationship+1+0.0)/(csr_matrices.size()+0.0)*RTX_3090_SM_NUM;
        num_sms_for_all_prev_relation_vect.push_back(num_sms_for_this_and_prev_relation);
    }
    for (int idx_relationship = 0; idx_relationship < csr_matrices.size(); idx_relationship++) {
        num_sms_for_same_relation_vect.push_back(num_sms_for_all_prev_relation_vect[idx_relationship+1] - num_sms_for_all_prev_relation_vect[idx_relationship]);
    }
    num_sms_for_all_prev_relation_vect.erase(num_sms_for_all_prev_relation_vect.begin());
    int idx_curr_relation = 0;
    for (int idx_sm = 0; idx_sm<RTX_3090_SM_NUM; idx_sm++) {
        if (idx_curr_relation< num_sms_for_all_prev_relation_vect.size()-1 && idx_sm>=num_sms_for_all_prev_relation_vect[idx_curr_relation]) {
            idx_curr_relation++;
        }
        smid_relation_id_vect.push_back(idx_curr_relation);
        beg_row_idxes_vect.push_back(0);
        num_sms_for_same_relation_per_sm_vect.push_back(num_sms_for_same_relation_vect[idx_curr_relation]);
    }


    dim3 block(RTX_3090_BLOCKSIZE, 1, 1);
    dim3 grid(RTX_3090_SM_NUM, 1, 1);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    EdgeSoftmaxMultiRelationsCSRSKernel<<<grid, block>>>(thrust::raw_pointer_cast(smid_relation_id_vect.data()), thrust::raw_pointer_cast(beg_row_idxes_vect.data()), thrust::raw_pointer_cast(num_sms_for_same_relation_per_sm_vect.data()), thrust::raw_pointer_cast(outNodes_per_relation_vect.data()), outEdges, nnzs_vect.size(), csr_matrices[0].num_rows, thrust::raw_pointer_cast(nnzs_vect.data()), thrust::raw_pointer_cast(matCols_vect.data()), thrust::raw_pointer_cast(matRows_vect.data()), edge_input_data, mus);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU EdgeSoftmaxMultiRelationsCSRSKernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    cudaFree(outEdges);
    cudaFree(edge_input_data);
    cudaFree(mus);

}


std::pair<std::pair<std::vector<int>,std::vector<int>>, std::vector<int>> generate_concatenate_coo_format(std::vector<std::vector<int>> coo_matrices_data){
    std::vector<std::pair<std::pair<int, int>, int>> coo_matrices_data_concatenated;
    
    
    for (int idx_matrix = 0; idx_matrix < coo_matrices_data.size(); idx_matrix++){
        for (int idx_element =0; idx_element<coo_matrices_data[idx_matrix].size()/2; idx_element++){
            coo_matrices_data_concatenated.push_back(std::make_pair(std::make_pair(coo_matrices_data[idx_matrix][idx_element], coo_matrices_data[idx_matrix][idx_element+coo_matrices_data[idx_matrix].size()/2]), idx_matrix));
            // result_data.push_back(idx_matrix);
            // result_row_indices.push_back(coo_matrices_data[idx_matrix][idx_element]);
            // result_col_indices.push_back(coo_matrices_data[idx_matrix][idx_element+coo_matrices_data[idx_matrix].size()/2]);
        }
    }

    std::sort(coo_matrices_data_concatenated.begin(), coo_matrices_data_concatenated.end(), [](const std::pair<std::pair<int, int>, int>& a, const std::pair<std::pair<int, int>, int>& b) {
        return a.first.first < b.first.first;
    });

    std::vector<int> result_row_indices;
    std::vector<int> result_col_indices;
    std::vector<int> result_data;
    for (int idx = 0; idx<coo_matrices_data_concatenated.size(); idx++){
        result_row_indices.push_back(coo_matrices_data_concatenated[idx].first.first);
        result_col_indices.push_back(coo_matrices_data_concatenated[idx].first.second);
        result_data.push_back(coo_matrices_data_concatenated[idx].second);
    }
    return std::make_pair(std::make_pair(result_row_indices, result_col_indices), result_data);
}

int main()
{
    

    
    
    std::vector<unsigned long> written_by_shape;
    std::vector<unsigned long> has_shape;
    std::vector<unsigned long> is_about_shape;
    std::vector<unsigned long> cited_shape;
    std::vector<unsigned long> citing_shape;
    std::vector<unsigned long> writing_shape;

    bool fortran_order;
    std::vector<int> written_by_data;
    std::vector<int> has_data;
    std::vector<int> is_about_data;
    std::vector<int> cited_data;
    std::vector<int> citing_data;
    std::vector<int> writing_data;

    npy::LoadArrayFromNumpy("data/written-by_coo_2.npy", written_by_shape, fortran_order, written_by_data);
    npy::LoadArrayFromNumpy("data/has_coo_2.npy", has_shape, fortran_order, has_data);
    npy::LoadArrayFromNumpy("data/is-about_coo_2.npy", is_about_shape, fortran_order, is_about_data);
    npy::LoadArrayFromNumpy("data/cited_coo_2.npy", cited_shape, fortran_order, cited_data);
    npy::LoadArrayFromNumpy("data/citing_coo_2.npy", citing_shape, fortran_order, citing_data);
    npy::LoadArrayFromNumpy("data/writing_coo_2.npy", writing_shape, fortran_order, writing_data);

   std::vector<int> max_idxes;
   max_idxes.push_back(*std::max_element(written_by_data.begin(), written_by_data.end()));
    max_idxes.push_back(*std::max_element(has_data.begin(), has_data.end()));
    max_idxes.push_back(*std::max_element(is_about_data.begin(), is_about_data.end()));
    max_idxes.push_back(*std::max_element(cited_data.begin(), cited_data.end()));
    max_idxes.push_back(*std::max_element(citing_data.begin(), citing_data.end()));
    max_idxes.push_back(*std::max_element(writing_data.begin(), writing_data.end()));
   int max_idx=*std::max_element(max_idxes.begin(), max_idxes.end());

   //cusp::csr_matrix<int, int, cusp::host_memory> csr_host(5, 8, 12);
   cusp::coo_matrix<int, int, cusp::host_memory> written_by_coo_h(max_idx+1, max_idx+1, written_by_data.size()/2);
   for (int idx = 0; idx < written_by_data.size() / 2; idx++) {
       written_by_coo_h.row_indices[idx]= written_by_data[idx];
       written_by_coo_h.column_indices[idx]=written_by_data[idx + written_by_data.size() / 2];
   }
   cusp::csr_matrix<int, int, cusp::host_memory> written_by_csr_h(written_by_coo_h);

   cusp::coo_matrix<int, int, cusp::host_memory> has_coo_h(max_idx+1, max_idx+1, has_data.size()/2);
   for (int idx = 0; idx < has_data.size() / 2; idx++) {
       has_coo_h.row_indices[idx]= has_data[idx];
       has_coo_h.column_indices[idx]=has_data[idx + has_data.size() / 2];
   }
   cusp::csr_matrix<int, int, cusp::host_memory> has_csr_h(has_coo_h);

   cusp::coo_matrix<int, int, cusp::host_memory> is_about_coo_h(max_idx+1, max_idx+1, is_about_data.size()/2);
   for (int idx = 0; idx < is_about_data.size() / 2; idx++) {
       is_about_coo_h.row_indices[idx]= is_about_data[idx];
       is_about_coo_h.column_indices[idx]=is_about_data[idx + is_about_data.size() / 2];
   }
   cusp::csr_matrix<int, int, cusp::host_memory> is_about_csr_h(is_about_coo_h);

   cusp::coo_matrix<int, int, cusp::host_memory> cited_coo_h(max_idx+1, max_idx+1, cited_data.size()/2);
   for (int idx = 0; idx < cited_data.size() / 2; idx++) {
       cited_coo_h.row_indices[idx]= cited_data[idx];
       cited_coo_h.column_indices[idx]=cited_data[idx + cited_data.size() / 2];
   }
   cusp::csr_matrix<int, int, cusp::host_memory> cited_csr_h(cited_coo_h);
   

   cusp::coo_matrix<int, int, cusp::host_memory> citing_coo_h(max_idx+1, max_idx+1, citing_data.size()/2);
   for (int idx = 0; idx < citing_data.size() / 2; idx++) {
       citing_coo_h.row_indices[idx]= citing_data[idx];
       citing_coo_h.column_indices[idx]=citing_data[idx + citing_data.size() / 2];
   }
   cusp::csr_matrix<int, int, cusp::host_memory> citing_csr_h(citing_coo_h);

   cusp::coo_matrix<int, int, cusp::host_memory> writing_coo_h(max_idx+1, max_idx+1, writing_data.size()/2);
   for (int idx = 0; idx < writing_data.size() / 2; idx++) {
       writing_coo_h.row_indices[idx]= writing_data[idx];
       writing_coo_h.column_indices[idx]=writing_data[idx + writing_data.size() / 2];
   }
   cusp::csr_matrix<int, int, cusp::host_memory> writing_csr_h(writing_coo_h);


   std::vector<std::vector<int>> coo_matrices_data={written_by_data, has_data, is_about_data, cited_data, citing_data, writing_data};
   std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<int>> concatenated_coo_result = generate_concatenate_coo_format(coo_matrices_data);
    std::vector<int> concatenated_coo_row_indices = concatenated_coo_result.first.first;
    std::vector<int> concatenated_coo_column_indices = concatenated_coo_result.first.second;
    std::vector<int> concatenated_coo_values = concatenated_coo_result.second;
    cusp::coo_matrix<int, int, cusp::host_memory> concatenated_coo_h(max_idx+1, max_idx+1, concatenated_coo_values.size());
    for (int idx = 0; idx < concatenated_coo_values.size(); idx++) {
        concatenated_coo_h.row_indices[idx]= concatenated_coo_row_indices[idx];
        concatenated_coo_h.column_indices[idx]=concatenated_coo_column_indices[idx];
        concatenated_coo_h.values[idx]=concatenated_coo_values[idx];
    }
    cusp::csr_matrix<int, int, cusp::host_memory> concatenated_csr_h(concatenated_coo_h);
    cusp::csr_matrix<int, int, cusp::device_memory> concatenated_csr_d(concatenated_csr_h);


    cusp::coo_matrix<int, int, cusp::device_memory> writing_coo_d(writing_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> has_coo_d(has_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> cited_coo_d(cited_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> is_about_coo_d(is_about_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> written_by_coo_d(written_by_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> citing_coo_d(citing_coo_h);

    cusp::csr_matrix<int, int, cusp::device_memory> written_by_csr_d(written_by_csr_h);
    cusp::csr_matrix<int, int, cusp::device_memory> has_csr_d(has_csr_h);
    cusp::csr_matrix<int, int, cusp::device_memory> is_about_csr_d(is_about_csr_h);
    cusp::csr_matrix<int, int, cusp::device_memory> cited_csr_d(cited_csr_h);
    cusp::csr_matrix<int, int, cusp::device_memory> citing_csr_d(citing_csr_h);
    cusp::csr_matrix<int, int, cusp::device_memory> writing_csr_d(writing_csr_h);

    // for (int idx = 0; idx < concatenated_coo_values.size(); idx++) {
    //     std::cout << concatenated_csr_h.values[idx] << ",";
    // }



    doGPUEdgeSoftmaxMultiRelationsCSRSKernel({ written_by_csr_d, has_csr_d, is_about_csr_d, cited_csr_d, citing_csr_d, writing_csr_d });
    doGPUEdgeSoftmaxConcatenatedCSRKernel(concatenated_csr_d, 6);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
   

    return 0;
}

