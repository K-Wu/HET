#include "hetero_edgesoftmax.h"
#include "EdgeSoftmax_1/EdgeSoftmax_1.h"
#include "EdgeSoftmax_4/EdgeSoftmax_4.h"
#include "EdgeAttention_4/EdgeAttention_4.h"
#include "MyHyb/MyHyb.h"


int main(){
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

    std::vector<cusp::csr_matrix<int, int, cusp::host_memory>> csr_matrices = {written_by_csr_h, has_csr_h, is_about_csr_h, cited_csr_h, citing_csr_h, writing_csr_h};

    int64_t total_num_nnzs = 0;
    for (auto& csr_matrix : csr_matrices)
    {
        total_num_nnzs += csr_matrix.values.size();
    }
    thrust::host_vector<int> eids(total_num_nnzs);
    thrust::sequence<>(eids.begin(), eids.end(), 0);
    MyHeteroSeparateCSR<int, std::allocator<int>> myHeteroSeparateCSR(csr_matrices, eids);
    //printf("myHeteroSeparateCSR.num_rows: %d\n", myHeteroSeparateCSR.num_rows);
    //printf("myHeteroSeparateCSR.num_cols: %d\n", myHeteroSeparateCSR.num_cols);
    //printf("myHeteroSeparateCSR.total_num_nnzs: %d\n", myHeteroSeparateCSR.total_num_nnzs);

    MyHeteroIntegratedCSR<int, std::allocator<int>> myHeteroIntegratedCSR = ToIntegratedCSR<std::allocator<int>, std::allocator<int>, int>(myHeteroSeparateCSR);
    MyHeteroIntegratedCSR<int, std::allocator<int>> myHeteroIntegratedCSR_transposed(myHeteroIntegratedCSR);
    myHeteroIntegratedCSR_transposed.Transpose();
    MyHeteroIntegratedCSR<int, std::allocator<int>> myHeteroIntegratedCSR2(myHeteroIntegratedCSR_transposed);
    myHeteroIntegratedCSR2.Transpose();
    assert(IsEqual<>(myHeteroIntegratedCSR,myHeteroIntegratedCSR2));
    
    MyHeteroSeparateCSR<int, std::allocator<int>> myHeteroSeparateCSR2 = ToSeparateCSR<std::allocator<int>, std::allocator<int>, int>(myHeteroIntegratedCSR);
    assert(IsEqual<>(myHeteroSeparateCSR, myHeteroSeparateCSR2));
    MyHyb<int, std::allocator<int>, MyHeteroSeparateCSR<int, std::allocator<int>>> myhyb=IntegratedCSRToHyb_CPU<int>(myHeteroIntegratedCSR, 5, 5, 10000);
    MyHyb<int, std::allocator<int>, MyHeteroSeparateCSR<int, std::allocator<int>>> myhyb2=IntegratedCSRToHyb_CPU<int>(myHeteroIntegratedCSR, 5, 7, 10000);
    myHeteroIntegratedCSR.VerifyThrustAllocatorRecognizable();
    myHeteroSeparateCSR.VerifyThrustAllocatorRecognizable();
    myHeteroSeparateCSR2.VerifyThrustAllocatorRecognizable();
    myhyb.VerifyThrustAllocatorRecognizable();
    myhyb2.VerifyThrustAllocatorRecognizable();
    assert(IsEqual<>(myhyb,myhyb2));

}