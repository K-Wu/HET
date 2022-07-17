#include "hetero_edgesoftmax.h"
#include "EdgeSoftmax_1/EdgeSoftmax_1.h"
#include "EdgeSoftmax_4/EdgeSoftmax_4.h"
#include "EdgeAttention_4/EdgeAttention_4.h"

struct identity_firstfloat
{
    // float4 argument_type;
    // float result_type;
    __thrust_exec_check_disable__
        __host__ __device__ const float &
        operator()(const float4 &x) const { return x.x; }
}; // end identity_firstfloat

struct compare_firstfloat
{
    __host__ __device__ bool operator()(float4 x, float y) const
    {
        return x.x == y;
    }
};

struct compare_float4
{
    __host__ __device__ bool operator()(float4 x, float4 y) const
    {
        return x.x == y.x && x.y == y.y && x.z == y.z && x.w == y.w;
    }
};

struct approx_compare_float4
{
    __host__ __device__ bool operator()(float4 x, float4 y) const
    {
        return (x.x - y.x) / (x.x + y.x) < 0.0001 && (x.y - y.y) / (x.y + y.y) < 0.0001 && (x.z - y.z) / (x.z + y.z) < 0.0001 && (x.w - y.w) / (x.w + y.w) < 0.0001;
    }
};

std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<int>> generate_concatenate_coo_format(std::vector<std::vector<int>> coo_matrices_data)
{
    std::vector<std::pair<std::pair<int, int>, int>> coo_matrices_data_concatenated;

    for (int idx_matrix = 0; idx_matrix < coo_matrices_data.size(); idx_matrix++)
    {
        for (int idx_element = 0; idx_element < coo_matrices_data[idx_matrix].size() / 2; idx_element++)
        {
            coo_matrices_data_concatenated.push_back(std::make_pair(std::make_pair(coo_matrices_data[idx_matrix][idx_element], coo_matrices_data[idx_matrix][idx_element + coo_matrices_data[idx_matrix].size() / 2]), idx_matrix));
            // result_data.push_back(idx_matrix);
            // result_row_indices.push_back(coo_matrices_data[idx_matrix][idx_element]);
            // result_col_indices.push_back(coo_matrices_data[idx_matrix][idx_element+coo_matrices_data[idx_matrix].size()/2]);
        }
    }

    std::sort(coo_matrices_data_concatenated.begin(), coo_matrices_data_concatenated.end(), [](const std::pair<std::pair<int, int>, int> &a, const std::pair<std::pair<int, int>, int> &b)
              { return a.first.first < b.first.first; });

    std::vector<int> result_row_indices;
    std::vector<int> result_col_indices;
    std::vector<int> result_data;
    for (int idx = 0; idx < coo_matrices_data_concatenated.size(); idx++)
    {
        result_row_indices.push_back(coo_matrices_data_concatenated[idx].first.first);
        result_col_indices.push_back(coo_matrices_data_concatenated[idx].first.second);
        result_data.push_back(coo_matrices_data_concatenated[idx].second);
    }
    return std::make_pair(std::make_pair(result_row_indices, result_col_indices), result_data);
}

template <typename Iterator>
void print_range(const std::string &name, Iterator first, Iterator last)
{
    // from thrust example
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": (" << std::distance(first, last) << ")";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
    std::cout << "\n";
}

int basic_correctness_test()
{

    std::vector<unsigned long> written_by_shape;
    std::vector<unsigned long> has_shape;
    std::vector<unsigned long> is_about_shape;
    std::vector<unsigned long> cited_shape;
    std::vector<unsigned long> citing_shape;
    std::vector<unsigned long> writing_shape;

    bool fortran_order = false;
    std::vector<int> written_by_data;
    std::vector<int> has_data;
    std::vector<int> is_about_data;
    std::vector<int> cited_data;
    std::vector<int> citing_data;
    std::vector<int> writing_data;

    // npy::LoadArrayFromNumpy("data/ogbn_mag/written-by_coo_1.npy", written_by_shape, fortran_order, written_by_data);
    // npy::LoadArrayFromNumpy("data/ogbn_mag/has_coo_1.npy", has_shape, fortran_order, has_data);
    // npy::LoadArrayFromNumpy("data/ogbn_mag/is-about_coo_1.npy", is_about_shape, fortran_order, is_about_data);
    // npy::LoadArrayFromNumpy("data/ogbn_mag/cited_coo_1.npy", cited_shape, fortran_order, cited_data);
    // npy::LoadArrayFromNumpy("data/ogbn_mag/citing_coo_1.npy", citing_shape, fortran_order, citing_data);
    // npy::LoadArrayFromNumpy("data/ogbn_mag/writing_coo_1.npy", writing_shape, fortran_order, writing_data);

    npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/written-by_coo_2.npy", written_by_shape, fortran_order, written_by_data);
    npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/has_coo_2.npy", has_shape, fortran_order, has_data);
    npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/is-about_coo_2.npy", is_about_shape, fortran_order, is_about_data);
    npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/cited_coo_2.npy", cited_shape, fortran_order, cited_data);
    npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/citing_coo_2.npy", citing_shape, fortran_order, citing_data);
    npy::LoadArrayFromNumpy("data/ogbn_mag_0.1/writing_coo_2.npy", writing_shape, fortran_order, writing_data);

    std::vector<int> max_idxes;
    max_idxes.push_back(*std::max_element(written_by_data.begin(), written_by_data.end()));
    max_idxes.push_back(*std::max_element(has_data.begin(), has_data.end()));
    max_idxes.push_back(*std::max_element(is_about_data.begin(), is_about_data.end()));
    max_idxes.push_back(*std::max_element(cited_data.begin(), cited_data.end()));
    max_idxes.push_back(*std::max_element(citing_data.begin(), citing_data.end()));
    max_idxes.push_back(*std::max_element(writing_data.begin(), writing_data.end()));
    int max_idx = *std::max_element(max_idxes.begin(), max_idxes.end());

    // cusp::csr_matrix<int, int, cusp::host_memory> csr_host(5, 8, 12);
    cusp::coo_matrix<int, int, cusp::host_memory> written_by_coo_h(max_idx + 1, max_idx + 1, written_by_data.size() / 2);
    for (int idx = 0; idx < written_by_data.size() / 2; idx++)
    {
        written_by_coo_h.row_indices[idx] = written_by_data[idx];
        written_by_coo_h.column_indices[idx] = written_by_data[idx + written_by_data.size() / 2];
    }

    cusp::coo_matrix<int, int, cusp::host_memory> has_coo_h(max_idx + 1, max_idx + 1, has_data.size() / 2);
    for (int idx = 0; idx < has_data.size() / 2; idx++)
    {
        has_coo_h.row_indices[idx] = has_data[idx];
        has_coo_h.column_indices[idx] = has_data[idx + has_data.size() / 2];
    }

    cusp::coo_matrix<int, int, cusp::host_memory> is_about_coo_h(max_idx + 1, max_idx + 1, is_about_data.size() / 2);
    for (int idx = 0; idx < is_about_data.size() / 2; idx++)
    {
        is_about_coo_h.row_indices[idx] = is_about_data[idx];
        is_about_coo_h.column_indices[idx] = is_about_data[idx + is_about_data.size() / 2];
    }

    cusp::coo_matrix<int, int, cusp::host_memory> cited_coo_h(max_idx + 1, max_idx + 1, cited_data.size() / 2);
    for (int idx = 0; idx < cited_data.size() / 2; idx++)
    {
        cited_coo_h.row_indices[idx] = cited_data[idx];
        cited_coo_h.column_indices[idx] = cited_data[idx + cited_data.size() / 2];
    }

    cusp::coo_matrix<int, int, cusp::host_memory> citing_coo_h(max_idx + 1, max_idx + 1, citing_data.size() / 2);
    for (int idx = 0; idx < citing_data.size() / 2; idx++)
    {
        citing_coo_h.row_indices[idx] = citing_data[idx];
        citing_coo_h.column_indices[idx] = citing_data[idx + citing_data.size() / 2];
    }

    cusp::coo_matrix<int, int, cusp::host_memory> writing_coo_h(max_idx + 1, max_idx + 1, writing_data.size() / 2);
    for (int idx = 0; idx < writing_data.size() / 2; idx++)
    {
        writing_coo_h.row_indices[idx] = writing_data[idx];
        writing_coo_h.column_indices[idx] = writing_data[idx + writing_data.size() / 2];
    }

    written_by_coo_h.sort_by_row_and_column();
    has_coo_h.sort_by_row_and_column();
    is_about_coo_h.sort_by_row_and_column();
    cited_coo_h.sort_by_row_and_column();
    citing_coo_h.sort_by_row_and_column();
    writing_coo_h.sort_by_row_and_column();

    cusp::csr_matrix<int, int, cusp::host_memory> written_by_csr_h(written_by_coo_h);
    cusp::csr_matrix<int, int, cusp::host_memory> has_csr_h(has_coo_h);
    cusp::csr_matrix<int, int, cusp::host_memory> is_about_csr_h(is_about_coo_h);
    cusp::csr_matrix<int, int, cusp::host_memory> cited_csr_h(cited_coo_h);
    cusp::csr_matrix<int, int, cusp::host_memory> citing_csr_h(citing_coo_h);
    cusp::csr_matrix<int, int, cusp::host_memory> writing_csr_h(writing_coo_h);

    std::vector<std::vector<int>> coo_matrices_data = {written_by_data, has_data, is_about_data, cited_data, citing_data, writing_data};
    std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<int>> concatenated_coo_result = generate_concatenate_coo_format(coo_matrices_data);
    std::vector<int> concatenated_coo_row_indices = concatenated_coo_result.first.first;
    std::vector<int> concatenated_coo_column_indices = concatenated_coo_result.first.second;
    std::vector<int> concatenated_coo_values = concatenated_coo_result.second;
    cusp::coo_matrix<int, int, cusp::host_memory> concatenated_coo_h(max_idx + 1, max_idx + 1, concatenated_coo_values.size());
    for (int idx = 0; idx < concatenated_coo_values.size(); idx++)
    {
        concatenated_coo_h.row_indices[idx] = concatenated_coo_row_indices[idx];
        concatenated_coo_h.column_indices[idx] = concatenated_coo_column_indices[idx];
        concatenated_coo_h.values[idx] = concatenated_coo_values[idx];
    }
    int total_nnzs = 0;
    for (int idx_matrix = 0; idx_matrix < coo_matrices_data.size(); idx_matrix++)
    {
        total_nnzs += coo_matrices_data[idx_matrix].size() / 2;
    }
    assert(total_nnzs == concatenated_coo_values.size());

    printf("concatenated coo edge %d node %d\n", total_nnzs, concatenated_coo_h.num_rows);

    concatenated_coo_h.sort_by_row_and_column();

    cusp::csr_matrix<int, int, cusp::host_memory> concatenated_csr_h(concatenated_coo_h);
    cusp::csr_matrix<int, int, cusp::device_memory> concatenated_csr_d(concatenated_csr_h);

    cusp::coo_matrix<int, int, cusp::device_memory> concatenated_coo_d(concatenated_coo_h);

    cusp::coo_matrix<int, int, cusp::device_memory> writing_coo_d(writing_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> has_coo_d(has_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> cited_coo_d(cited_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> is_about_coo_d(is_about_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> written_by_coo_d(written_by_coo_h);
    cusp::coo_matrix<int, int, cusp::device_memory> citing_coo_d(citing_coo_h);

    cusp::csr_matrix<int, int, cusp::device_memory> written_by_csr_d;
    cusp::csr_matrix<int, int, cusp::device_memory> has_csr_d;
    cusp::csr_matrix<int, int, cusp::device_memory> is_about_csr_d;
    cusp::csr_matrix<int, int, cusp::device_memory> cited_csr_d;
    cusp::csr_matrix<int, int, cusp::device_memory> citing_csr_d;
    cusp::csr_matrix<int, int, cusp::device_memory> writing_csr_d;

    cusp::convert(written_by_coo_d, written_by_csr_d);
    cusp::convert(has_coo_d, has_csr_d);
    cusp::convert(cited_coo_d, cited_csr_d);
    cusp::convert(is_about_coo_d, is_about_csr_d);
    cusp::convert(citing_coo_d, citing_csr_d);
    cusp::convert(writing_coo_d, writing_csr_d);

    cusp::csr_matrix<int, int, cusp::device_memory> written_by_csc_d;
    cusp::csr_matrix<int, int, cusp::device_memory> has_csc_d;
    cusp::csr_matrix<int, int, cusp::device_memory> is_about_csc_d;
    cusp::csr_matrix<int, int, cusp::device_memory> cited_csc_d;
    cusp::csr_matrix<int, int, cusp::device_memory> citing_csc_d;
    cusp::csr_matrix<int, int, cusp::device_memory> writing_csc_d;
    cusp::csr_matrix<int, int, cusp::device_memory> concatenated_csc_d;

    cusp::transpose(written_by_csr_d, written_by_csc_d);
    cusp::transpose(has_csr_d, has_csc_d);
    cusp::transpose(is_about_csr_d, is_about_csc_d);
    cusp::transpose(cited_csr_d, cited_csc_d);
    cusp::transpose(citing_csr_d, citing_csc_d);
    cusp::transpose(writing_csr_d, writing_csc_d);
    cusp::transpose(concatenated_csr_d, concatenated_csc_d);

    // In the given dataset, all the edges are sampled, so it isn't necessarily true that the inverse relation's csr and csc should match.

    // assert(thrust::equal(thrust::device, written_by_csr_d.row_offsets.begin(),written_by_csr_d.row_offsets.end(), writing_csc_d.row_offsets.begin()));
    // assert(thrust::equal(thrust::device, cited_csr_d.row_offsets.begin(),cited_csr_d.row_offsets.end(), citing_csc_d.row_offsets.begin()));
    //  print_range("written_by_csr_d.row_offsets", written_by_csr_d.row_offsets.begin(), written_by_csr_d.row_offsets.end());
    //  print_range("written_by_csr_d.column_indices", written_by_csr_d.column_indices.begin(), written_by_csr_d.column_indices.end());
    //  print_range("written_by_coo_d.row_indices", written_by_coo_d.row_indices.begin(), written_by_coo_d.row_indices.end());
    //  print_range("written_by_coo_d.column_indices", written_by_coo_d.column_indices.begin(), written_by_coo_d.column_indices.end());
    //  print_range("written_by_csc_d.row_offsets", written_by_csc_d.row_offsets.begin(), written_by_csc_d.row_offsets.end());
    //  print_range("written_by_csc_d.column_indices", written_by_csc_d.column_indices.begin(), written_by_csc_d.column_indices.end());
    //  print_range("writing_csc_d.row_offsets", writing_csc_d.row_offsets.begin(), writing_csc_d.row_offsets.end());
    //  print_range("writing_csc_d.column_indices", writing_csc_d.column_indices.begin(), writing_csc_d.column_indices.end());
    //  print_range("writing_coo_d.row_offsets", writing_coo_d.row_indices.begin(), writing_coo_d.row_indices.end());
    //  print_range("writing_coo_d.column_indices", writing_coo_d.column_indices.begin(), writing_coo_d.column_indices.end());
    //  print_range("writing_csr_d.row_offsets", writing_csr_d.row_offsets.begin(), writing_csr_d.row_offsets.end());
    //  print_range("writing_csr_d.column_indices", writing_csr_d.column_indices.begin(), writing_csr_d.column_indices.end());
    // assert(thrust::equal(thrust::device, written_by_csr_d.column_indices.begin(), written_by_csr_d.column_indices.end(), writing_csc_d.column_indices.begin()));
    // assert(thrust::equal(thrust::device, cited_csr_d.column_indices.begin(), cited_csr_d.column_indices.end(), citing_csc_d.column_indices.begin()));

    // for (int idx = 0; idx < concatenated_coo_values.size(); idx++) {
    //     std::cout << concatenated_csr_h.values[idx] << ",";
    // }
    std::vector<thrust::device_vector<float>> MultiCSRoutNodes_per_relation_vect_vect = doGPUEdgeSoftmaxMultiCSRsKernel({written_by_csr_d, has_csr_d, is_about_csr_d, cited_csr_d, citing_csr_d, writing_csr_d}, false);
    std::vector<thrust::device_vector<float>> CSRoutNodes_per_relation_vect_vect = doGPUEdgeSoftmaxConcatenatedCSRKernel(concatenated_csr_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false);
    // std::vector<thrust::device_vector<float>> MultiCSRoutNodes_per_relation_vect_vect = doGPUEdgeSoftmaxMultiCSRsKernel({written_by_csr_d, has_csr_d, is_about_csr_d, cited_csr_d, citing_csr_d, writing_csr_d}, false);
    std::vector<thrust::device_vector<float>> CSCoutNodes_per_relation_vect_vect = doGPUEdgeSoftmaxConcatenatedCSCKernel(concatenated_csc_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false);
    std::vector<thrust::device_vector<float>> COOoutNodes_per_relation_vect_vect = doGPUEdgeSoftmaxConcatenatedCOOKernel(concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false);
    std::vector<thrust::device_vector<float4>> COOoutNodes_4_per_relation_vect_vect = doGPUEdgeSoftmax_4ConcatenatedCOOKernel(concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false);

    std::vector<thrust::device_vector<float>> MultiCSCoutNodes_per_relation_vect_vect = doGPUEdgeSoftmaxMultiCSCsKernel({written_by_csc_d, has_csc_d, is_about_csc_d, cited_csc_d, citing_csc_d, writing_csc_d}, false);
    std::vector<thrust::device_vector<float>> MultiCOOoutNodes_per_relation_vect_vect = doGPUEdgeSoftmaxMultiCOOsKernel<cusp::coo_matrix<int, int, cusp::device_memory>>({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, false);

    thrust::device_vector<float4> COOOutEdgeAttention_per_relation = doGPUEdgeAttentionConcatenatedCOOKernel_128_16({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false);
    thrust::device_vector<float4> COOOutEdgeAttention_per_relation_128_8 = doGPUEdgeAttentionConcatenatedCOOKernel_128_8({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false);
    thrust::device_vector<float4> COOOutEdgeAttention_per_relation_256_8 = doGPUEdgeAttentionConcatenatedCOOKernel_256_8({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false, false);
    thrust::device_vector<float4> COOOutEdgeAttention_per_relation_256_8_2 = doGPUEdgeAttentionConcatenatedCOOKernel_256_8({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false, true);
    thrust::device_vector<float4> COOOutEdgeAttention_per_relation_256_32 = doGPUEdgeAttentionConcatenatedCOOKernel_256_32({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false, false);
    thrust::device_vector<float4> COOOutEdgeAttention_per_relation_256_32_2 = doGPUEdgeAttentionConcatenatedCOOKernel_256_32({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false, true);
    thrust::device_vector<float4> COOOutEdgeAttention_per_relation_512_32 = doGPUEdgeAttentionConcatenatedCOOKernel_512_32({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false, false);
    thrust::device_vector<float4> COOOutEdgeAttention_per_relation_512_32_A100 = doGPUEdgeAttentionConcatenatedCOOKernel_512_32_A100({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false, false);

    thrust::device_vector<float4> COOOutEdgeAttention_per_relation_512_32_asyncmemcpy = doGPUEdgeAttentionConcatenatedCOOKernel_512_32_asyncmemcpy({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false, false);

    // thrust::device_vector<float4> COOOutEdgeAttention_per_relation_512_32_2 = doGPUEdgeAttentionConcatenatedCOOKernel_512_32({written_by_coo_d, has_coo_d, is_about_coo_d, cited_coo_d, citing_coo_d, writing_coo_d}, concatenated_coo_d, MultiCSRoutNodes_per_relation_vect_vect.size(), false, true);

    for (int idx = 0; idx < MultiCSRoutNodes_per_relation_vect_vect.size(); idx++)
    {
        thrust::host_vector<float> MultiCSRoutNodes_curr_relation_vect_h(MultiCSRoutNodes_per_relation_vect_vect[idx]);
        thrust::host_vector<float> CSRoutNodes_curr_relation_vect_h(CSRoutNodes_per_relation_vect_vect[idx]);
        // thrust::negate<float> op;
        /*for (int idx_node = 0; idx_node < CSRoutNodes_curr_relation_vect_h.size(); idx_node++) {
            std::cout << CSRoutNodes_curr_relation_vect_h[idx_node] << ",";
        }
        std::cout << std::endl;
        for (int idx_node = 0; idx_node < MultiCSRoutNodes_curr_relation_vect_h.size(); idx_node++) {
            std::cout << MultiCSRoutNodes_curr_relation_vect_h[idx_node] << ",";
        }
        std::cout << std::endl;*/
        /*for (int idx_node = 0; idx_node < CSRoutNodes_curr_relation_vect_h.size(); idx_node++) {
            std::cout << CSRoutNodes_curr_relation_vect_h[idx_node]- MultiCSRoutNodes_curr_relation_vect_h[idx_node] << ",";
            if (idx_node % 32 == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;*/
        std::cout << thrust::raw_pointer_cast(MultiCSRoutNodes_per_relation_vect_vect[idx].data()) << "," << thrust::raw_pointer_cast(CSRoutNodes_per_relation_vect_vect[idx].data()) << std::endl;
        std::cout << MultiCSRoutNodes_per_relation_vect_vect[idx].size() << "," << CSRoutNodes_per_relation_vect_vect[idx].size() << std::endl;
        /*thrust::transform(thrust::device,MultiCSRoutNodes_per_relation_vect_vect[idx].begin(), MultiCSRoutNodes_per_relation_vect_vect[idx].end(), MultiCSRoutNodes_per_relation_vect_vect[idx].begin(), op);*/
        std::cout << thrust::equal(thrust::device, MultiCSRoutNodes_per_relation_vect_vect[idx].begin(), MultiCSRoutNodes_per_relation_vect_vect[idx].end(), CSRoutNodes_per_relation_vect_vect[idx].begin());
        std::cout << thrust::equal(thrust::device, CSCoutNodes_per_relation_vect_vect[idx].begin(), CSCoutNodes_per_relation_vect_vect[idx].end(), CSRoutNodes_per_relation_vect_vect[idx].begin());
        std::cout << thrust::equal(thrust::device, MultiCSCoutNodes_per_relation_vect_vect[idx].begin(), MultiCSCoutNodes_per_relation_vect_vect[idx].end(), CSRoutNodes_per_relation_vect_vect[idx].begin());
        std::cout << thrust::equal(thrust::device, MultiCOOoutNodes_per_relation_vect_vect[idx].begin(), MultiCOOoutNodes_per_relation_vect_vect[idx].end(), CSRoutNodes_per_relation_vect_vect[idx].begin());
        std::cout << thrust::equal(thrust::device, COOoutNodes_per_relation_vect_vect[idx].begin(), COOoutNodes_per_relation_vect_vect[idx].end(), CSRoutNodes_per_relation_vect_vect[idx].begin());

        // thrust::transform(COOoutNodes_4_per_relation_vect_vect[idx].begin(), COOoutNodes_4_per_relation_vect_vect[idx].end(), COOoutNodes_per_relation_vect_vect[idx].begin(), identity_firstfloat());
        std::cout << thrust::equal(thrust::device, COOoutNodes_4_per_relation_vect_vect[idx].begin(), COOoutNodes_4_per_relation_vect_vect[idx].end(), COOoutNodes_per_relation_vect_vect[idx].begin(), compare_firstfloat());

        // print_range("MultiCSRoutNodes_per_relation_vect_vect[idx]", MultiCSRoutNodes_per_relation_vect_vect[idx].begin(), MultiCSRoutNodes_per_relation_vect_vect[idx].end());
        //  print_range("CSRoutNodes_per_relation_vect_vect[idx]", CSRoutNodes_per_relation_vect_vect[idx].begin(), CSRoutNodes_per_relation_vect_vect[idx].end());
        // print_range("COOoutNodes_per_relation_vect_vect[idx]", COOoutNodes_per_relation_vect_vect[idx].begin(), COOoutNodes_per_relation_vect_vect[idx].end());
        std::cout << std::endl;
    }
    std::cout << "COOOutEdgeAttention_per_relation" << std::endl;
    std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_128_8.begin(), COOOutEdgeAttention_per_relation_128_8.end(), COOOutEdgeAttention_per_relation.begin(), compare_float4());
    std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_256_8.begin(), COOOutEdgeAttention_per_relation_256_8.end(), COOOutEdgeAttention_per_relation.begin(), compare_float4());
    std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_256_8_2.begin(), COOOutEdgeAttention_per_relation_256_8_2.end(), COOOutEdgeAttention_per_relation.begin(), compare_float4());

    std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_256_32.begin(), COOOutEdgeAttention_per_relation_256_32.end(), COOOutEdgeAttention_per_relation.begin(), compare_float4());
    std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_256_32_2.begin(), COOOutEdgeAttention_per_relation_256_32_2.end(), COOOutEdgeAttention_per_relation.begin(), compare_float4());
    std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_512_32.begin(), COOOutEdgeAttention_per_relation_512_32.end(), COOOutEdgeAttention_per_relation.begin(), compare_float4());
    std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_512_32_asyncmemcpy.begin(), COOOutEdgeAttention_per_relation_512_32_asyncmemcpy.end(), COOOutEdgeAttention_per_relation.begin(), compare_float4());
    std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_512_32_A100.begin(), COOOutEdgeAttention_per_relation_512_32_A100.end(), COOOutEdgeAttention_per_relation.begin(), approx_compare_float4());

    // std::cout << thrust::equal(thrust::device, COOOutEdgeAttention_per_relation_512_32_2.begin(), COOOutEdgeAttention_per_relation_512_32_2.end(), COOOutEdgeAttention_per_relation.begin(),compare_float4());

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    return 0;
}

int main()
{
    return basic_correctness_test();
}
