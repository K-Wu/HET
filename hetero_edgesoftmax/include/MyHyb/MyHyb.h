#pragma once
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cusp/csr_matrix.h>
#include <map>
#include <set>
#include <optional>
#include <numeric>
//#include <optional>

#ifdef MyHyb_NONEXISTENT_ELEMENT
#error "MyHyb_DEFINE_CONSTANT is already defined"
#else 
#define MyHyb_NONEXISTENT_ELEMENT -1
#endif 


template <typename IdxType>
thrust::host_vector<IdxType> TransposeCSR(thrust::detail::vector_base<IdxType, std::allocator<IdxType>>& row_ptr, thrust::detail::vector_base<IdxType, std::allocator<IdxType>>& col_idx ){
    //transpose the csr matrix, and return the permutation array so that rel_type in integratedCSR and eid in FusedGAT can be mapped to the new order using the permutation array.
    thrust::host_vector<IdxType> permutation(col_idx.size());
    thrust::sequence<>(permutation.begin(),permutation.end(), 0);
    thrust::host_vector<IdxType> new_row_ptr(row_ptr.size());
    thrust::host_vector<IdxType> new_col_idx(col_idx.size());

    std::map<IdxType, std::vector<std::pair<IdxType, IdxType>>> col_row_map;

    for (int64_t i = 0; i < row_ptr.size() - 1; i++) {
        for (int64_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            assert(col_idx[j]<row_ptr.size()-1);
            col_row_map[col_idx[j]].push_back(std::make_pair(i, permutation[j]));
        }
    }

    new_row_ptr[0] = 0;
    for (int64_t idxNode = 0; idxNode< row_ptr.size() - 1; idxNode++) { // assert num_rows == num_cols
        new_row_ptr[idxNode+1] = new_row_ptr[idxNode] + col_row_map[idxNode].size();
        for (int64_t idxEdgeForCurrNode = 0; idxEdgeForCurrNode< col_row_map[idxNode].size(); idxEdgeForCurrNode++) {
            new_col_idx[new_row_ptr[idxNode] + idxEdgeForCurrNode] = col_row_map[idxNode][idxEdgeForCurrNode].first;
            permutation[new_row_ptr[idxNode] + idxEdgeForCurrNode] = col_row_map[idxNode][idxEdgeForCurrNode].second;
        }
    }

    thrust::copy(new_row_ptr.begin(), new_row_ptr.end(), row_ptr.begin());
    thrust::copy(new_col_idx.begin(), new_col_idx.end(), col_idx.begin());
    return permutation;

}

template <typename IdxType, typename Alloc>
bool IsDeviceVector(const thrust::detail::vector_base<IdxType, Alloc>& vec)
{
    //return std::is_same<thrust::device_vector<IdxType>::Parent::storage_type::allocator_type,Alloc>::value;
    return std::is_same<typename thrust::device_vector<IdxType>::allocator_type,Alloc>::value;
}

template <typename IdxType, typename Alloc>
bool IsHostVector(const thrust::detail::vector_base<IdxType, Alloc>& vec)
{
    //return std::is_same<thrust::host_vector<IdxType>::Parent::storage_type::allocator_type,Alloc>::value;
    return std::is_same<typename thrust::host_vector<IdxType>::allocator_type,Alloc>::value;
}


template<typename IdxType, typename Alloc>
class MyHeteroIntegratedCSR;

template<typename IdxType, typename Alloc>
class MyHeteroSeparateCSR{
public:
    int64_t num_rows;
    int64_t num_cols;
    int64_t num_rels;
    std::vector<int64_t> num_nnzs;
    int total_num_nnzs;
    //thrust::detail::vector_base<IdxType, Alloc> rel_ptr;
    thrust::detail::vector_base<IdxType, Alloc> row_ptr;
    thrust::detail::vector_base<IdxType, Alloc> col_idx;

    MyHeteroSeparateCSR() = default;
    //std::allocator<T>
    //thrust::device_allocator<T>
    template<typename OtherAlloc>
    MyHeteroSeparateCSR(const int64_t num_rows, const int64_t num_cols, const int64_t num_rels, const std::vector<int64_t>& num_nnzs,
                            /*const thrust::detail::vector_base<IdxType, OtherAlloc>& rel_ptr,*/
                           const thrust::detail::vector_base<IdxType, OtherAlloc>& row_ptr,
                           const thrust::detail::vector_base<IdxType, OtherAlloc>& col_idx){
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->num_rels = num_rels;
        this->num_nnzs = num_nnzs;
        this->total_num_nnzs = std::reduce(num_nnzs.begin(), num_nnzs.end());
        //this->rel_ptr = rel_ptr;
        this->row_ptr = row_ptr;
        this->col_idx = col_idx;
    }

    template<typename ValueType, typename MemorySpace>
    MyHeteroSeparateCSR(const std::vector<cusp::csr_matrix<IdxType, ValueType, MemorySpace>>& cusp_csrs){
        this->num_nnzs = std::vector<int64_t>(cusp_csrs.size(),0);
        for (int64_t csr_idx=0;csr_idx< cusp_csrs.size();csr_idx++){
            this->num_rows = std::max(this->num_rows, (int64_t)cusp_csrs[csr_idx].num_rows);
            this->num_cols = std::max(this->num_cols, (int64_t)cusp_csrs[csr_idx].num_cols);
            this->num_nnzs[csr_idx]=cusp_csrs[csr_idx].num_entries;
        }
        this->num_rels = cusp_csrs.size();
        this->total_num_nnzs = std::reduce(num_nnzs.begin(), num_nnzs.end());
        this->row_ptr = thrust::detail::vector_base<IdxType, Alloc>(this->num_rels*this->num_rows+1,0);
        this->col_idx = thrust::detail::vector_base<IdxType, Alloc>(this->total_num_nnzs,0);

        for (int64_t IdxRelationship = 0; IdxRelationship < this->num_rels; IdxRelationship++){
            for (int64_t IdxRow =0; IdxRow<this->num_rows; IdxRow++){
                if (cusp_csrs[IdxRelationship].row_offsets.size()<=IdxRow){
                    // this csr has less rows than the current row index
                    this->row_ptr[IdxRow+IdxRelationship*this->num_rows+1]=this->row_ptr[IdxRow+IdxRelationship*this->num_rows]; 
                }
                else{
                    int64_t NumEdgesFromThisRowAndRelationship = cusp_csrs[IdxRelationship].row_offsets[IdxRow+1]-cusp_csrs[IdxRelationship].row_offsets[IdxRow];
                    this->row_ptr[IdxRow+IdxRelationship*this->num_rows+1]=this->row_ptr[IdxRow+IdxRelationship*this->num_rows]+NumEdgesFromThisRowAndRelationship;
                    assert(this->row_ptr[IdxRow+IdxRelationship*this->num_rows+1]==this->row_ptr[IdxRelationship*this->num_rows]+cusp_csrs[IdxRelationship].row_offsets[IdxRow+1]);
                }
            }
            assert(this->row_ptr[(1+IdxRelationship)*this->num_rows]== std::reduce(num_nnzs.begin(), std::next(num_nnzs.begin(),IdxRelationship+1)));
            for (int64_t IdxEdgeThisRelationship = 0; IdxEdgeThisRelationship<cusp_csrs[IdxRelationship].num_entries;IdxEdgeThisRelationship++){
                this->col_idx[this->row_ptr[IdxRelationship*this->num_rows]+IdxEdgeThisRelationship]=cusp_csrs[IdxRelationship].column_indices[IdxEdgeThisRelationship];
            }                        
        }
    }

    template<typename OtherType, typename OtherAlloc>
    void Transpose(std::optional<typename std::reference_wrapper<typename thrust::detail::vector_base<OtherType, OtherAlloc>>> eids){
        thrust::host_vector<IdxType>  permutation = TransposeCSR(row_ptr, col_idx);
        assert(num_rows==num_cols);
        if (eids.has_value()){
            thrust::detail::vector_base<IdxType, OtherAlloc>& eids_ref = eids.value().get();
            thrust::detail::vector_base<OtherType, OtherAlloc> new_eids(eids_ref.size());
            thrust::detail::vector_base<IdxType, OtherAlloc> eids_new(permutation.size());
            typedef typename thrust::detail::vector_base<IdxType, OtherAlloc>::iterator ElementIterator;
            typedef typename thrust::host_vector<IdxType>::iterator IndexIterator;
            thrust::permutation_iterator<ElementIterator, IndexIterator> permute_iter(eids_ref.begin(), permutation.begin());
            //for (int64_t idx = 0; idx < eids_ref.size(); idx++) {
            //    new_eids[idx] = *permute_iter;
            //    permute_iter++;
            //}
            thrust::copy(permute_iter, permute_iter + eids_ref.size(), new_eids.begin());
            thrust::copy(new_eids.begin(), new_eids.end(), eids_ref.begin());
        }
    }

    template<typename OtherAlloc>
    MyHeteroSeparateCSR(const MyHeteroSeparateCSR<IdxType, OtherAlloc>& csr){
        //this->rel_ptr = csr.rel_ptr;
        this->row_ptr = csr.row_ptr;
        this->col_idx = csr.col_idx;
        this->num_nnzs = csr.num_nnzs;
        this->num_rows = csr.num_rows;
        this->num_cols = csr.num_cols;
        this->num_rels = csr.num_rels;
        this->total_num_nnzs = csr.total_num_nnzs;
    }
    void VerifyThrustAllocatorRecognizable() const {
        // the allocator should be either std::allocator (thrust::host_vector's) or thrust::device_allocator (thrust::device_vector's)
        assert(IsHostVector(row_ptr) || IsDeviceVector(row_ptr));
        //assert(std::is_same(thrust::device_vector<IdxType>::Parent::storage_type::allocator_type,Alloc)||
        //std::is_same(thrust::host_vector<IdxType>::Parent::storage_type::allocator_type,Alloc));
    }
    bool IsDataOnGPU() const {
        return IsDeviceVector(row_ptr);
    }
    bool IsDataOnCPU() const {
        return IsHostVector(row_ptr);
    }
};

template<typename IdxType, typename Alloc>
class MyHeteroIntegratedCSR{
public:
    MyHeteroIntegratedCSR() = default;

    int64_t num_rows;
    int64_t num_cols;
    int64_t num_rels;
    std::vector<int64_t> num_nnzs;
    int64_t total_num_nnzs;
    thrust::detail::vector_base<IdxType, Alloc> row_ptr;
    thrust::detail::vector_base<IdxType, Alloc> col_idx;
    thrust::detail::vector_base<IdxType, Alloc> rel_type;
    thrust::detail::vector_base<IdxType, Alloc> eids;
    template<typename OtherAlloc>
    MyHeteroIntegratedCSR(const int64_t num_rows, const int64_t num_cols, const int64_t num_rels,
                            const std::vector<int64_t>& num_nnzs, 
                            const thrust::detail::vector_base<IdxType, OtherAlloc>& row_ptr,
                            const thrust::detail::vector_base<IdxType, OtherAlloc>& col_idx,
                            const thrust::detail::vector_base<IdxType, OtherAlloc>& rel_type,
                            const thrust::detail::vector_base<IdxType, OtherAlloc>& eids){
        this->total_num_nnzs = std::reduce(num_nnzs.begin(), num_nnzs.end());
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->num_rels = num_rels;
        this->num_nnzs = num_nnzs;
        this->row_ptr = row_ptr;
        this->col_idx = col_idx;
        this->rel_type = rel_type;
        this->eids = eids;
    }

    MyHeteroIntegratedCSR(const thrust::detail::vector_base<IdxType, std::allocator<IdxType>>& row_ptr,
                            const thrust::detail::vector_base<IdxType, std::allocator<IdxType>>& col_idx,
                            const thrust::detail::vector_base<IdxType, std::allocator<IdxType>>& rel_type,
                            const thrust::detail::vector_base<IdxType, std::allocator<IdxType>>& eids){
        this->row_ptr = row_ptr;
        this->col_idx = col_idx;
        this->rel_type = rel_type;
        this->num_rows = row_ptr.size()-1;
        this->num_cols = this->num_rows;
        // num rels is the largest index in rel_type
        this->num_rels = (*std::max_element(rel_type.begin(), rel_type.end()))+1;
        this->total_num_nnzs = col_idx.size();
        // count rel_type to get num_nnz of each type
        std::vector<int64_t> num_nnz_type(this->num_rels,0);
        for (int64_t i=0;i<this->total_num_nnzs;i++){
            num_nnz_type[rel_type[i]]++;
        }
        this->num_nnzs = num_nnz_type;
        this->eids = eids;
    }

    template<typename OtherAlloc>
    MyHeteroIntegratedCSR(const MyHeteroIntegratedCSR<IdxType, OtherAlloc>& csr){
        this->rel_type = csr.rel_type;
        this->row_ptr = csr.row_ptr;
        this->col_idx = csr.col_idx;
        this->eids = csr.eids;
        this->num_nnzs = csr.num_nnzs;
        this->num_rows = csr.num_rows;
        this->num_cols = csr.num_cols;
        this->num_rels = csr.num_rels;
        this->total_num_nnzs = csr.total_num_nnzs;
    }

    void SortByEdgeType_CPU() {
        assert(IsDataOnCPU());
        for (int64_t IdxRow =0; IdxRow < num_rows; IdxRow++){
            std::vector<std::pair<IdxType, std::pair<IdxType,IdxType>>> EdgeRelationshipPairFromThisNode;
            for (IdxType IdxEdge = row_ptr[IdxRow]; IdxEdge < row_ptr[IdxRow+1]; IdxEdge++){
                IdxType IdxSrcNode = IdxRow;
                IdxType IdxDestNode = col_idx[IdxEdge];
                IdxType IdxRelationshipEdge = rel_type[IdxEdge];
                IdxType IdxEdgeID = eids[IdxEdge];
                //rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(IdxDestNode);
                EdgeRelationshipPairFromThisNode.push_back(std::make_pair(IdxDestNode, std::make_pair(IdxRelationshipEdge, IdxEdgeID)));
            }
            std::sort(EdgeRelationshipPairFromThisNode.begin(), EdgeRelationshipPairFromThisNode.end(), [](const std::pair<IdxType, std::pair<IdxType,IdxType>>& a, const std::pair<IdxType, std::pair<IdxType,IdxType>>& b){
                return a.second.first < b.second.first;
            });
            //write back
            for (int64_t IdxEdge = 0; IdxEdge < EdgeRelationshipPairFromThisNode.size(); IdxEdge++){
                col_idx[row_ptr[IdxRow]+IdxEdge] = EdgeRelationshipPairFromThisNode[IdxEdge].first;
                rel_type[row_ptr[IdxRow]+IdxEdge] = EdgeRelationshipPairFromThisNode[IdxEdge].second.first;
                eids[row_ptr[IdxRow]+IdxEdge] = EdgeRelationshipPairFromThisNode[IdxEdge].second.second;
            }
        }
    }

    //template<typename OtherType, typename OtherAlloc>
    void Transpose(/*std::optional<typename std::reference_wrapper<typename thrust::detail::vector_base<OtherType, OtherAlloc>>> eids*/){
        assert(num_rows==num_cols);


        thrust::host_vector<IdxType>  permutation = TransposeCSR(row_ptr, col_idx);
        
        // if (eids.has_value()){
            
        //     thrust::detail::vector_base<IdxType, OtherAlloc>& eids_ref = eids.value().get();
        //     thrust::detail::vector_base<OtherType, OtherAlloc> new_eids(eids_ref.size());
        //     thrust::detail::vector_base<IdxType, OtherAlloc> eids_new(permutation.size());
        //     typedef typename thrust::detail::vector_base<IdxType, OtherAlloc>::iterator ElementIterator;
        //     typedef typename thrust::host_vector<IdxType>::iterator IndexIterator;
        //     thrust::permutation_iterator<ElementIterator, IndexIterator> permute_iter(eids_ref.begin(), permutation.begin());
        //     thrust::copy(permute_iter, permute_iter+ eids_ref.size(), new_eids.begin());
        //     thrust::copy(new_eids.begin(), new_eids.end(), eids_ref.begin());
        // }

        // work on rel_types
        typedef typename thrust::detail::vector_base<IdxType, Alloc>::iterator ElementIterator;
        typedef typename thrust::host_vector<IdxType>::iterator IndexIterator;
        thrust::detail::vector_base<IdxType, Alloc> new_rel_types(permutation.size());
        thrust::permutation_iterator<ElementIterator, IndexIterator> permute_iter(rel_type.begin(), permutation.begin());
        thrust::copy(permute_iter, permute_iter + permutation.size(), new_rel_types.begin());
        thrust::copy(new_rel_types.begin(), new_rel_types.end(), rel_type.begin());

        //work on eids
        thrust::detail::vector_base<IdxType, Alloc> new_eids(permutation.size());
        thrust::permutation_iterator<ElementIterator, IndexIterator> permute_iter_eids(this->eids.begin(), permutation.begin());
        thrust::copy(permute_iter_eids, permute_iter_eids + permutation.size(), new_eids.begin());
        thrust::copy(new_eids.begin(), new_eids.end(), this->eids.begin());
    }

    bool IsSortedByEdgeType_CPU() {
        assert(IsDataOnCPU());
        for (int64_t IdxRow =0; IdxRow < num_rows; IdxRow++){
            std::vector<std::pair<IdxType, IdxType>> EdgeRelationshipPairFromThisNode;
            for (IdxType IdxEdge = row_ptr[IdxRow]; IdxEdge < row_ptr[IdxRow+1]; IdxEdge++){
                IdxType IdxSrcNode = IdxRow;
                IdxType IdxDestNode = col_idx[IdxEdge];
                IdxType IdxRelationshipEdge = rel_type[IdxEdge];
                //rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(IdxDestNode);
                EdgeRelationshipPairFromThisNode.push_back(std::make_pair(IdxDestNode, IdxRelationshipEdge));
            }
            std::sort(EdgeRelationshipPairFromThisNode.begin(), EdgeRelationshipPairFromThisNode.end(), [](const std::pair<IdxType, IdxType>& a, const std::pair<IdxType, IdxType>& b){
                return a.second < b.second;
            });

            //check if the input csr is sorted
            for (int64_t IdxEdge = 0; IdxEdge < EdgeRelationshipPairFromThisNode.size(); IdxEdge++){
                if (col_idx[row_ptr[IdxRow]+IdxEdge] != EdgeRelationshipPairFromThisNode[IdxEdge].first){
                    return false;
                }

                if (rel_type[row_ptr[IdxRow]+IdxEdge] = EdgeRelationshipPairFromThisNode[IdxEdge].second){
                    return false;
                }
            }
        }
        return true;
    }



    void SortByEdgeType_CPU(const thrust::detail::vector_base<IdxType, Alloc>& eids) {
        assert(IsDataOnCPU());
        for (int64_t IdxRow =0; IdxRow < num_rows; IdxRow++){
            std::vector<std::pair<std::pair<IdxType, IdxType>, IdxType>> EdgeRelationshipEidsTupleFromThisNode;
            for (IdxType IdxEdge = row_ptr[IdxRow]; IdxEdge < row_ptr[IdxRow+1]; IdxEdge++){
                IdxType IdxSrcNode = IdxRow;
                IdxType IdxDestNode = col_idx[IdxEdge];
                IdxType IdxRelationshipEdge = rel_type[IdxEdge];
                IdxType ElementEids = eids[IdxEdge]; 
                //rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(IdxDestNode);
                EdgeRelationshipEidsTupleFromThisNode.push_back(std::make_pair(std::make_pair(IdxDestNode, IdxRelationshipEdge),ElementEids));
            }
            std::sort(EdgeRelationshipEidsTupleFromThisNode.begin(), EdgeRelationshipEidsTupleFromThisNode.end(), [](const std::pair<IdxType, IdxType>& a, const std::pair<IdxType, IdxType>& b){
                return a.first.second < b.first.second;
            });
            //write back
            for (int64_t IdxEdge = 0; IdxEdge < EdgeRelationshipEidsTupleFromThisNode.size(); IdxEdge++){
                col_idx[row_ptr[IdxRow]+IdxEdge] = EdgeRelationshipEidsTupleFromThisNode[IdxEdge].first.first;
                rel_type[row_ptr[IdxRow]+IdxEdge] = EdgeRelationshipEidsTupleFromThisNode[IdxEdge].first.second;
                eids[row_ptr[IdxRow]+IdxEdge] = EdgeRelationshipEidsTupleFromThisNode[IdxEdge].second;
            }
        }
    }

    void VerifyThrustAllocatorRecognizable() const {
        // the allocator should be either std::allocator (thrust::host_vector's) or thrust::device_allocator (thrust::device_vector's)
        assert(IsHostVector(rel_type) || IsDeviceVector(rel_type));
        //assert(std::is_same(thrust::device_vector<IdxType>::Parent::storage_type::allocator_type,Alloc)||
        //std::is_same(thrust::host_vector<IdxType>::Parent::storage_type::allocator_type,Alloc));
    }

    bool IsDataOnGPU() const {
        return IsDeviceVector(rel_type);
    }
    bool IsDataOnCPU() const {
        return IsHostVector(rel_type);
    }
};


template<typename IdxType, typename Alloc, typename CSRType>
class MyHyb{
    //[0,HybIndexMax] has both ELL and CSR format
    //(HybIndexMax, num_rows) has only CSR format
public:

    int64_t HybIndexMax;
    int64_t ELL_logical_width;
    int64_t ELL_physical_width; // logical width and actual width is similar to lda in BLAS routines.
    int64_t num_rows;
    int64_t num_cols;
    int64_t num_rels;
    std::vector<int64_t> num_nnzs;
    int64_t total_num_nnzs;
    CSRType csr;
    thrust::detail::vector_base<IdxType, Alloc> ELLColIdx;
    thrust::detail::vector_base<IdxType, Alloc> ELLRelType;
    thrust::detail::vector_base<IdxType, Alloc> ELLEids;

    MyHyb(const int64_t HybIndexMax, const int64_t ELL_logical_width, const int64_t ELL_physical_width, const CSRType& csr,
    const thrust::detail::vector_base<IdxType, Alloc>& ELLColIdx,
    const thrust::detail::vector_base<IdxType, Alloc>& ELLRelType,
    const thrust::detail::vector_base<IdxType, Alloc>& ELLEids,
    const int64_t num_rows, const int64_t num_cols, const int64_t num_rels,
    const std::vector<int64_t>& num_nnzs){
        this->HybIndexMax = HybIndexMax;
        this->ELL_logical_width = ELL_logical_width;
        this->ELL_physical_width = ELL_physical_width;
        this->ELLEids = ELLEids;
        this->csr = csr;
        this->ELLColIdx = ELLColIdx;
        this->ELLRelType = ELLRelType;
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->num_rels = num_rels;
        this->num_nnzs=num_nnzs;
        this->total_num_nnzs = std::reduce(num_nnzs.begin(), num_nnzs.end());
    }

    template<typename OtherAlloc, typename OtherCSRType>
    MyHyb(const MyHyb<IdxType, OtherAlloc, OtherCSRType>& another_myhyb) {
        this->HybIndexMax = another_myhyb.HybIndexMax;
        this->ELL_logical_width = another_myhyb.ELL_logical_width;
        this->ELL_physical_width = another_myhyb.ELL_physical_width;
        this->ELLEids = another_myhyb.ELLEids;
        this->csr = another_myhyb.csr;
        this->ELLColIdx = another_myhyb.ELLColIdx;
        this->ELLRelType = another_myhyb.ELLRelType;
        this->num_rows = another_myhyb.num_rows;
        this->num_cols = another_myhyb.num_cols;
        this->num_rels = another_myhyb.num_rels;
        this->num_nnzs=another_myhyb.num_nnzs;
        this->total_num_nnzs = another_myhyb.total_num_nnzs;
    }


    void VerifyThrustAllocatorRecognizable() const {
        // the allocator should be either std::allocator (thrust::host_vector's) or thrust::device_allocator (thrust::device_vector's)
        assert(IsHostVector(ELLRelType) || IsDeviceVector(ELLRelType));
        //assert(std::is_same(thrust::device_vector<IdxType>::Parent::storage_type::allocator_type,Alloc)||
        //std::is_same(thrust::host_vector<IdxType>::Parent::storage_type::allocator_type,Alloc));
    }

    bool IsDataOnGPU() const {
        return IsDeviceVector(ELLRelType);
    }
    bool IsDataOnCPU() const {
        return IsHostVector(ELLRelType);
    }

};

// separate CSR to integrated CSR
template <typename IdxType>
//MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>> 
MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>> ToIntegratedCSR_GPU(const MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>& csr){
    //csr.rel_ptr;
    //csr.row_ptr;
    //csr.col_idx;
    thrust::device_vector<IdxType> result_row_ptr;
    thrust::device_vector<IdxType> result_col_idx;
    thrust::device_vector<IdxType> result_rel_type;
    // TODO: implement here
    assert(0 && "GPU kernel not implemented");
    MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>> result(csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptr, result_col_idx, result_rel_type);
    //return dummy to avoid compiler error
    return result;
}

template <typename IdxType>
MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> ToIntegratedCSR_CPU(const MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>& csr){
    //csr.rel_ptr;
    //csr.row_ptr;
    //csr.col_idx;
    int64_t total_num_nnzs = std::reduce(csr.num_nnzs.begin(), csr.num_nnzs.end());

    std::vector<std::map<IdxType,  std::vector<IdxType>>> edge_to_rel_type(csr.num_rows);

    thrust::host_vector<IdxType> result_row_ptr(csr.num_rows+1, 0);
    thrust::host_vector<IdxType> result_col_idx(total_num_nnzs, 0);
    thrust::host_vector<IdxType> result_rel_type(total_num_nnzs, 0);

    
    for (int64_t IdxRelation = 0; IdxRelation < csr.num_rels; IdxRelation++){
        for (int64_t IdxRow = 0; IdxRow < csr.num_rows; IdxRow++){
            for (IdxType IdxElementColIdxWithOffset = csr.row_ptr[IdxRelation*csr.num_rows + IdxRow]; IdxElementColIdxWithOffset < csr.row_ptr[IdxRelation*csr.num_rows + IdxRow+1]; IdxElementColIdxWithOffset++){
                IdxType IdxSrcNode = IdxRow;
                IdxType IdxDestNode = csr.col_idx[IdxElementColIdxWithOffset];
                IdxType IdxRelationshipEdge = IdxRelation;
                edge_to_rel_type[IdxSrcNode][IdxDestNode].push_back(IdxRelationshipEdge);
            }
        }
    }

    IdxType currEdgeIdx = 0;
    for (int64_t IdxRow = 0; IdxRow < csr.num_rows; IdxRow++){
        for (auto it = edge_to_rel_type[IdxRow].begin(); it != edge_to_rel_type[IdxRow].end(); it++){
            for (IdxType IdxElement = 0; IdxElement < it->second.size(); IdxElement++){
                result_col_idx[currEdgeIdx] = it->first;
                result_rel_type[currEdgeIdx] = it->second[IdxElement];
                currEdgeIdx+=1;
                result_row_ptr[IdxRow+1]+=1;
            }
        }
        result_row_ptr[IdxRow+1] = result_row_ptr[IdxRow] + result_row_ptr[IdxRow+1];
        assert(result_row_ptr[IdxRow+1] == currEdgeIdx);
    }

    MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> result(csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, result_row_ptr, result_col_idx, result_rel_type);

    return result;
}


// integrated CSR to separate CSR
template <typename IdxType>
//MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>> 
MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>> ToSeparateCSR_GPU(const MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>>& csr){
    //csr.row_ptr;
    //csr.col_idx;
    //csr.rel_type;
    //thrust::device_vector<IdxType> result_rel_type;
    thrust::device_vector<IdxType> result_row_ptr;
    thrust::device_vector<IdxType> result_col_idx;
    // TODO: implement here
    assert(0 && "GPU kernel not implemented");




    MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>> result(csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, /*result_rel_ptr,*/ result_row_ptr, result_col_idx);
    //return dummy to avoid compiler error
    return result;
}

template <typename IdxType>
bool IsEqual(const MyHeteroIntegratedCSR<IdxType, typename thrust::host_vector<IdxType>::allocator_type>& csr1, const MyHeteroIntegratedCSR<IdxType, typename thrust::host_vector<IdxType>::allocator_type>& csr2){
    if (csr1.num_rows != csr2.num_rows){
        return false;
    }
    if (csr1.num_cols != csr2.num_cols){
        return false;
    }
    if (csr1.num_rels != csr2.num_rels){
        return false;
    }
    if (csr1.total_num_nnzs != csr2.total_num_nnzs){
        return false;
    }
    if (csr1.num_nnzs.size() != csr2.num_nnzs.size()){
        return false;
    }
    for (int64_t idx_relationship = 0; idx_relationship < csr1.num_nnzs.size(); idx_relationship++){
        if (csr1.num_nnzs[idx_relationship] != csr2.num_nnzs[idx_relationship]){
            return false;
        }
    }

    //thrust::detail::vector_base<IdxType, Alloc> row_ptr;
    //thrust::detail::vector_base<IdxType, Alloc> col_idx;
    //thrust::detail::vector_base<IdxType, Alloc> rel_type;
    for (IdxType IdxRow = 0; IdxRow < csr1.num_rows; IdxRow++){
        if (csr1.row_ptr[IdxRow+1] != csr2.row_ptr[IdxRow+1]){
            return false;
        }
    }

    for (int64_t IdxRow = 0; IdxRow < csr1.num_rows; IdxRow++){
        std::set<std::pair<IdxType, IdxType>> DestNodeRelationshipPairs;
        for (int64_t IdxEdge = csr1.row_ptr[IdxRow]; IdxEdge < csr1.row_ptr[IdxRow+1] ; IdxEdge++){
            DestNodeRelationshipPairs.insert(std::make_pair<>(csr1.col_idx[IdxEdge], csr1.rel_type[IdxEdge]));
            DestNodeRelationshipPairs.insert(std::make_pair<>(csr2.col_idx[IdxEdge], csr2.rel_type[IdxEdge]));
        }
        // if there is any difference between the two CSRs, the size of the set will be larger than the difference of one csr row ptr
        if (DestNodeRelationshipPairs.size()!=csr2.row_ptr[IdxRow+1] - csr2.row_ptr[IdxRow]){
            return false;
        }

    }
    return true;
}

template <typename IdxType>
bool IsEqual(const MyHeteroSeparateCSR<IdxType, typename thrust::host_vector<IdxType>::allocator_type>& csr1, const MyHeteroSeparateCSR<IdxType, typename thrust::host_vector<IdxType>::allocator_type>& csr2){
    if (csr1.num_rows != csr2.num_rows){
        return false;
    }
    if (csr1.num_cols != csr2.num_cols){
        return false;
    }
    if (csr1.num_rels != csr2.num_rels){
        return false;
    }
    if (csr1.total_num_nnzs != csr2.total_num_nnzs){
        return false;
    }
    if (csr1.num_nnzs.size() != csr2.num_nnzs.size()){
        return false;
    }
    for (int64_t idx_relationship = 0; idx_relationship < csr1.num_nnzs.size(); idx_relationship++){
        if (csr1.num_nnzs[idx_relationship] != csr2.num_nnzs[idx_relationship]){
            return false;
        }
    }
    //thrust::detail::vector_base<IdxType, Alloc> rel_ptr;
    //thrust::detail::vector_base<IdxType, Alloc> row_ptr;
    //thrust::detail::vector_base<IdxType, Alloc> col_idx;

    for(int64_t IdxRelationship = 0; IdxRelationship < csr1.num_rels; IdxRelationship++){
        for (int64_t IdxRow = 0; IdxRow < csr1.num_rows; IdxRow++){
            if (csr1.row_ptr[IdxRelationship* csr1.num_rows + IdxRow + 1] != csr2.row_ptr[IdxRelationship* csr1.num_rows + IdxRow + 1]){
                return false;
            }
        }
    }

    for (int64_t IdxRowWithRelationship = 0; IdxRowWithRelationship < csr1.num_rows*csr1.num_rels;IdxRowWithRelationship++ ){
        std::set<IdxType> DestNode;
        for (int64_t IdxEdge = csr1.row_ptr[IdxRowWithRelationship]; IdxEdge < csr1.row_ptr[IdxRowWithRelationship+1] ; IdxEdge++){
            DestNode.insert(csr1.col_idx[IdxEdge]);
            DestNode.insert(csr2.col_idx[IdxEdge]);
        }
        // if there is any difference between the two CSRs, the size of the set will be larger than the difference of one csr row ptr
        if (DestNode.size()!=csr2.row_ptr[IdxRowWithRelationship+1] - csr2.row_ptr[IdxRowWithRelationship]){
            return false;
        }
    }
    return true;
}

template <typename IdxType>
MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> ToSeparateCSR_CPU(const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>& csr){
    //csr.row_ptr;
    //csr.col_idx;
    //csr.rel_type;
    //thrust::host_vector<IdxType> result_rel_ptr(csr.num_rels+1, 0);
    thrust::host_vector<IdxType> result_row_ptr(csr.num_rows*csr.num_rels+1, 0);
    thrust::host_vector<IdxType> result_col_idx(csr.total_num_nnzs, 0);
    // TODO: implement here
    std::vector<std::vector<std::vector<IdxType>>> rel_type_to_edges(csr.num_rels,std::vector<std::vector<IdxType>>(csr.num_rows,std::vector<IdxType>()));

    for (int64_t IdxRow =0; IdxRow < csr.num_rows; IdxRow++){
        for (IdxType IdxEdge = csr.row_ptr[IdxRow]; IdxEdge < csr.row_ptr[IdxRow+1]; IdxEdge++){
            IdxType IdxSrcNode = IdxRow;
            IdxType IdxDestNode = csr.col_idx[IdxEdge];
            IdxType IdxRelationshipEdge = csr.rel_type[IdxEdge];
            rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(IdxDestNode);
        }
    }
    for (int64_t IdxRelationship =0; IdxRelationship < csr.num_rels; IdxRelationship++){
        for (IdxType IdxRow = 0; IdxRow <csr.num_rows; IdxRow++ ){
            result_row_ptr[IdxRelationship*csr.num_rows+IdxRow+1]= result_row_ptr[IdxRelationship*csr.num_rows+IdxRow];
            for (IdxType IdxElement = 0; IdxElement < rel_type_to_edges[IdxRelationship][IdxRow].size(); IdxElement++){
                result_col_idx[result_row_ptr[IdxRelationship*csr.num_rows+IdxRow+1]] = rel_type_to_edges[IdxRelationship][IdxRow][IdxElement];
                result_row_ptr[IdxRelationship*csr.num_rows+IdxRow+1]+=1;
            }
        }
        //result_rel_ptr[IdxRelationship+1] = result_rel_ptr[IdxRelationship] + result_row_ptr[IdxRelationship*csr.num_rows+csr.num_rows];
    }

    MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> result(csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, /*result_rel_ptr,*/ result_row_ptr, result_col_idx);
    return result;
}


template <typename IdxType, typename CSRType>
bool IsEqual(const MyHyb<IdxType, std::allocator<IdxType>, CSRType>& myhyb1, const MyHyb<IdxType, std::allocator<IdxType>, CSRType>& myhyb2){
    //TODO: update with eids
    if (myhyb1.num_rows != myhyb2.num_rows){
        return false;
    }
    if (myhyb1.num_cols != myhyb2.num_cols){
        return false;
    }
    if (myhyb1.num_rels != myhyb2.num_rels){
        return false;
    }
    if (myhyb1.total_num_nnzs != myhyb2.total_num_nnzs){
        return false;
    }
    if (myhyb1.num_nnzs.size() != myhyb2.num_nnzs.size()){
        return false;
    }
    for (int64_t IdxRelationship = 0; IdxRelationship < myhyb1.num_rels; IdxRelationship++){
        if (myhyb1.num_nnzs[IdxRelationship] != myhyb2.num_nnzs[IdxRelationship]){
            return false;
        }
    }
    if (myhyb1.HybIndexMax != myhyb2.HybIndexMax){
        return false;
    }
    if (myhyb1.ELL_logical_width != myhyb2.ELL_logical_width){
        return false;
    }
    // ELL physical width could be different
    for (int64_t IdxNode = 0; IdxNode<myhyb1.HybIndexMax; IdxNode++){
        std::set<std::pair<IdxType,IdxType>> DestNodeRelTypeSet;
        for (int64_t IdxElement = IdxNode*myhyb1.ELL_physical_width; IdxElement < IdxNode*myhyb1.ELL_physical_width+myhyb1.ELL_logical_width; IdxElement++){
            IdxType IdxDestNode = myhyb1.ELLColIdx[IdxElement];
            IdxType IdxRelationship = myhyb1.ELLRelType[IdxElement];
            if (IdxDestNode == MyHyb_NONEXISTENT_ELEMENT){
                continue;
            }
            DestNodeRelTypeSet.insert(std::make_pair(IdxDestNode, IdxRelationship));
        }
        int64_t NumEdgesFromThisSourceNode1 = DestNodeRelTypeSet.size();
        for (int64_t IdxElement = IdxNode*myhyb2.ELL_physical_width; IdxElement < IdxNode*myhyb2.ELL_physical_width+myhyb2.ELL_logical_width; IdxElement++){
            IdxType IdxDestNode = myhyb2.ELLColIdx[IdxElement];
            IdxType IdxRelationship = myhyb2.ELLRelType[IdxElement];
            if (IdxDestNode == MyHyb_NONEXISTENT_ELEMENT){
                continue;
            }
            DestNodeRelTypeSet.insert(std::make_pair(IdxDestNode, IdxRelationship));
        }
        // check if the number of edges from this source node is the same
        if (DestNodeRelTypeSet.size() != NumEdgesFromThisSourceNode1){
            return false;
        }
    }

    return IsEqual(myhyb1.csr, myhyb2.csr);
}

template <typename IdxType>
MyHyb<IdxType, std::allocator<IdxType>, MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>> IntegratedCSRToHyb_CPU(const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>& csr, int64_t ELL_logical_width, int64_t ELL_physical_width, int64_t ELLMaxIndex){
    //csr.row_ptr;
    //csr.col_idx;
    //csr.rel_type;
    //TODO: implement here
    thrust::host_vector<IdxType> ELLColIdx(ELLMaxIndex*ELL_physical_width, MyHyb_NONEXISTENT_ELEMENT);
    thrust::host_vector<IdxType> ELLRelType(ELLMaxIndex*ELL_physical_width, MyHyb_NONEXISTENT_ELEMENT);
    //based on MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> ToSeparateCSR_CPU(const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>& csr)

    thrust::host_vector<IdxType> result_row_ptr(csr.num_rows*csr.num_rels+1, 0);
    thrust::host_vector<IdxType> result_col_idx(csr.total_num_nnzs, 0);
    // TODO: implement here

    std::vector<std::vector<std::vector<IdxType>>> residue_csr_rel_type_to_edges(csr.num_rels,std::vector<std::vector<IdxType>>(csr.num_rows,std::vector<IdxType>()));

    for (int64_t IdxRow =0; IdxRow < csr.num_rows; IdxRow++){
        for (IdxType IdxEdge = csr.row_ptr[IdxRow]; IdxEdge < csr.row_ptr[IdxRow+1]; IdxEdge++){
            
            IdxType IdxSrcNode = IdxRow;
            IdxType IdxDestNode = csr.col_idx[IdxEdge];
            IdxType IdxRelationshipEdge = csr.rel_type[IdxEdge];
            if (IdxEdge<ELL_logical_width && IdxRow<ELLMaxIndex){
                // store the edge in the ELL
                    ELLColIdx[IdxRow*ELL_physical_width+IdxEdge] = IdxDestNode;
                    ELLRelType[IdxRow*ELL_physical_width+IdxEdge] = IdxRelationshipEdge;
            }
            else{
                // store the rest into the CSR
                residue_csr_rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(IdxDestNode);
            }
        }
    }

    int64_t csr_total_num_nnz = 0;
    for (int64_t IdxRelationship =0; IdxRelationship < csr.num_rels; IdxRelationship++){
        for (IdxType IdxRow = 0; IdxRow <csr.num_rows; IdxRow++ ){
            result_row_ptr[IdxRelationship*csr.num_rows+IdxRow+1]= result_row_ptr[IdxRelationship*csr.num_rows+IdxRow];
            for (IdxType IdxElement = 0; IdxElement < residue_csr_rel_type_to_edges[IdxRelationship][IdxRow].size(); IdxElement++){
                result_col_idx[result_row_ptr[IdxRelationship*csr.num_rows+IdxRow+1]] = residue_csr_rel_type_to_edges[IdxRelationship][IdxRow][IdxElement];
                result_row_ptr[IdxRelationship*csr.num_rows+IdxRow+1]+=1;
                csr_total_num_nnz+=1;
            }
        }
        //result_rel_ptr[IdxRelationship+1] = result_rel_ptr[IdxRelationship] + result_row_ptr[IdxRelationship*csr.num_rows+csr.num_rows];
    }

    MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> resultCSR(csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, /*result_rel_ptr,*/ result_row_ptr, result_col_idx);
    MyHyb<IdxType, std::allocator<IdxType>, MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>> result_hyb(ELLMaxIndex, ELL_logical_width, ELL_physical_width, resultCSR, ELLColIdx, ELLRelType,  csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs);
    return result_hyb;
}

template <typename IdxType>
MyHyb<IdxType, std::allocator<IdxType>, MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>> IntegratedCSRToHyb_ADHOC_CPU(const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>& csr, int64_t ELL_logical_width, int64_t ELL_physical_width, int64_t ELLMaxIndex){
    // TODO: this is an ad hoc solution
    //csr.row_ptr;
    //csr.col_idx;
    //csr.rel_type;
    //TODO: implement here
    thrust::host_vector<IdxType> ELLColIdx(ELLMaxIndex*ELL_physical_width, MyHyb_NONEXISTENT_ELEMENT);
    thrust::host_vector<IdxType> ELLRelType(ELLMaxIndex*ELL_physical_width, MyHyb_NONEXISTENT_ELEMENT);
    thrust::host_vector<IdxType> ELLEids(ELLMaxIndex*ELL_physical_width, MyHyb_NONEXISTENT_ELEMENT);
    //based on MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> ToSeparateCSR_CPU(const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>& csr)

    thrust::host_vector<IdxType> result_row_ptr(csr.num_rows+1, 0);
    thrust::host_vector<IdxType> result_col_idx(csr.total_num_nnzs, 0);
    thrust::host_vector<IdxType> result_rel_type(csr.total_num_nnzs, 0);
    thrust::host_vector<IdxType> result_eids(csr.total_num_nnzs, 0);
    // TODO: implement here

    //std::vector<std::vector<std::vector<IdxType>>> residue_csr_rel_type_to_edges(csr.num_rels,std::vector<std::vector<IdxType>>(csr.num_rows,std::vector<IdxType>()));
    std::vector<std::vector<std::pair<IdxType, std::pair<IdxType, IdxType>>>> residue_csr_src_nodes_to_dests_with_rel_type_and_eids(csr.num_rows);
    for (int64_t IdxRow =0; IdxRow < csr.num_rows; IdxRow++){
        for (IdxType IdxEdge = csr.row_ptr[IdxRow]; IdxEdge < csr.row_ptr[IdxRow+1]; IdxEdge++){
            
            IdxType IdxSrcNode = IdxRow;
            IdxType IdxDestNode = csr.col_idx[IdxEdge];
            IdxType IdxRelationshipEdge = csr.rel_type[IdxEdge];
            IdxType IdxEid = csr.eids[IdxEdge];
            if (IdxEdge<ELL_logical_width && IdxRow<ELLMaxIndex){
                // store the edge in the ELL
                    ELLColIdx[IdxRow*ELL_physical_width+IdxEdge] = IdxDestNode;
                    ELLRelType[IdxRow*ELL_physical_width+IdxEdge] = IdxRelationshipEdge;
                    ELLEids[IdxRow*ELL_physical_width+IdxEdge] = IdxEid;
            }
            else{
                // store the rest into the CSR
                //residue_csr_rel_type_to_edges[IdxRelationshipEdge][IdxSrcNode].push_back(IdxDestNode);
                residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxSrcNode].push_back(std::make_pair(IdxDestNode, std::make_pair(IdxRelationshipEdge, IdxEid)));
            }
        }
    }

    int64_t csr_total_num_nnz = 0;
    
    for (IdxType IdxRow = 0; IdxRow <csr.num_rows; IdxRow++ ){
        result_row_ptr[IdxRow+1]= result_row_ptr[IdxRow];
        for (IdxType IdxElement = 0; IdxElement < residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxRow].size(); IdxElement++){
        
            result_col_idx[result_row_ptr[IdxRow+1]] = residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxRow][IdxElement].first;
            result_rel_type[result_row_ptr[IdxRow+1]] = residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxRow][IdxElement].second.first;
            result_eids[result_row_ptr[IdxRow+1]] = residue_csr_src_nodes_to_dests_with_rel_type_and_eids[IdxRow][IdxElement].second.second;
            result_row_ptr[csr.num_rows+IdxRow+1]+=1;
            csr_total_num_nnz+=1;
        }
    }
        

    MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>> resultCSR(csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, /*result_rel_ptr,*/ result_row_ptr, result_col_idx, result_rel_type, result_eids);
    MyHyb<IdxType, std::allocator<IdxType>, MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>> result_hyb(ELLMaxIndex, ELL_logical_width, ELL_physical_width, resultCSR, ELLColIdx, ELLRelType,ELLEids,  csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs);
    return result_hyb;
}



template <typename IdxType>
//MyHyb<IdxType, std::allocator<IdxType>, MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>>> 
MyHyb<IdxType, thrust::device_allocator<IdxType>, MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>> IntegratedCSRToHyb_GPU(const MyHeteroIntegratedCSR<IdxType, thrust::device_allocator<IdxType>>& csr, int64_t ELL_logical_width, int64_t ELL_physical_width, int64_t ELLMaxIndex){
    //csr.row_ptr;
    //csr.col_idx;
    //csr.rel_type;
    thrust::device_vector<IdxType> ELLColIdx(ELLMaxIndex*ELL_physical_width, MyHyb_NONEXISTENT_ELEMENT);
    thrust::device_vector<IdxType> ELLRelType(ELLMaxIndex*ELL_physical_width, MyHyb_NONEXISTENT_ELEMENT);
    //based on MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>> ToSeparateCSR_CPU(const MyHeteroIntegratedCSR<IdxType, std::allocator<IdxType>>& csr)

    thrust::device_vector<IdxType> result_row_ptr(csr.num_rows*csr.num_rels+1, 0);
    thrust::device_vector<IdxType> result_col_idx(csr.num_nnzs, 0);

    //TODO: implement here
    assert(0 && "GPU kernel not implemented");
    //return dummy to avoid compiler error
    MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>> resultCSR(csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs, /*result_rel_ptr,*/ result_row_ptr, result_col_idx);
    MyHyb<IdxType, thrust::device_allocator<IdxType>, MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>> result_hyb(ELLMaxIndex, ELL_logical_width, ELL_physical_width, resultCSR,  csr.num_rows, csr.num_cols, csr.num_rels, csr.num_nnzs);
    return result_hyb;

}

template <typename IdxType>
//MyHyb<IdxType, std::allocator<IdxType>, MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>> 
void SeparateCSRToHyb_CPU(const MyHeteroSeparateCSR<IdxType, std::allocator<IdxType>>& csr){
    //csr.rel_ptr;
    //csr.row_ptr;
    //csr.col_idx;
    //TODO: implement here
    assert(0 && "SeparateCSR to Hyb kernel not implemented");

}

template <typename IdxType>
//MyHyb<IdxType, std::allocator<IdxType>, MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>> 
void SeparateCSRToHyb_GPU(const MyHeteroSeparateCSR<IdxType, thrust::device_allocator<IdxType>>& csr){
    //csr.rel_ptr;
    //csr.row_ptr;
    //csr.col_idx;
    //TODO: implement here
    assert(0 && "SeparateCSR to Hyb kernel not implemented");
    //return dummy to avoid compiler error
}



template<typename InputAlloc, typename ReturnAlloc, typename IdxType>
MyHeteroIntegratedCSR<IdxType, ReturnAlloc> ToIntegratedCSR(const MyHeteroSeparateCSR<IdxType, InputAlloc>& csr){
    // get cpu and gpu implementation.
    if (csr.IsDataOnCPU()){
        auto ret_value =  ToIntegratedCSR_CPU<IdxType>(csr);
        return ret_value; // implicit type conversion here to match return ReturnAlloc type.
    }
    else{
        assert(csr.IsDataOnGPU());
        auto ret_value = ToIntegratedCSR_GPU<IdxType>(csr);
        return ret_value; // implicit type conversion here to match return ReturnAlloc type. 
    }
}

template<typename InputAlloc, typename ReturnAlloc, typename IdxType>
MyHeteroSeparateCSR<IdxType, ReturnAlloc> ToSeparateCSR(const MyHeteroIntegratedCSR<IdxType, InputAlloc>& csr){
    // get cpu and gpu implementation.
    if (csr.IsDataOnCPU()){
        auto ret_value =  ToSeparateCSR_CPU<IdxType>(csr);
        return ret_value; // implicit type conversion here to match return ReturnAlloc type.
    }
    else{
        assert(csr.IsDataOnGPU());
        auto ret_value = ToSeparateCSR_GPU<IdxType>(csr);
        return ret_value; // implicit type conversion here to match return ReturnAlloc type. 
    }
}



template<typename InputAlloc, typename ReturnAlloc, typename IdxType>
MyHyb<IdxType, ReturnAlloc, MyHeteroSeparateCSR<IdxType, ReturnAlloc>> MyHybIntegratedCSRToSeparateCSR(const MyHyb<IdxType, ReturnAlloc, MyHeteroIntegratedCSR<IdxType, ReturnAlloc>>& hyb) {
    MyHyb<IdxType, ReturnAlloc, MyHeteroSeparateCSR<IdxType, ReturnAlloc>> result(hyb.HybIndexMax, hyb.ELL_logical_width, hyb.ELL_physical_width,
        ToSeparateCSR<InputAlloc, ReturnAlloc, IdxType>(hyb.csr), hyb.ELLColIdx, hyb.ELLRelType, hyb.num_rows, hyb.num_cols, hyb.num_rels, hyb.num_nnzs);
    return result;
}