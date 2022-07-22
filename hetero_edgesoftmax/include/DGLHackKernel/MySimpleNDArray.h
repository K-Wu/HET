#pragma once
#include "DGLHackKernel.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <thrust/random.h>

// TODO: implement transpose (probably using permutation functionality if provided by thrust) and padding.
template <typename DType, typename Alloc>
class MySimpleNDArray{
public:
    thrust::detail::vector_base<DType, Alloc> data;
    std::vector<int64_t> shape;
    MySimpleNDArray(std::vector<int64_t> shape) : shape(shape) {
        int64_t element_num = 1;
        for (int64_t i = 0; i < shape.size(); i++) {
            element_num *= shape[i];
        }
        data.resize(element_num);
    }
    MySimpleNDArray() = default;
    template<typename OtherAlloc>
    MySimpleNDArray(std::vector<int64_t> shape, thrust::detail::vector_base<DType, OtherAlloc> data){
        this->shape = shape;
        this->data = data;
    }
    template<typename OtherAlloc>
    MySimpleNDArray(thrust::detail::vector_base<DType, OtherAlloc> data){
        this->data = data;
        this->shape = {data.size()};
    }
    template<typename OtherAlloc>
    MySimpleNDArray(const MySimpleNDArray<DType, OtherAlloc>& other) : shape(other.shape), data(other.data) {}

    /*! \return the data pointer with type. */
    //template <typename T>
    //inline T* Ptr() {
    //    return static_cast<T*>(thrust::raw_pointer_cast(data.data()));
    //}

    inline DType* Ptr() {
           return static_cast<DType*>(thrust::raw_pointer_cast(data.data()));
        }

    //template <typename T>
    //inline const T* Ptr() const {
    //    return static_cast<const T*>(thrust::raw_pointer_cast(data.data()));
    //}

    template <typename OtherAlloc>
    bool IsEqual(const MySimpleNDArray<DType, OtherAlloc>& other) const {
        if (shape.size() != other.shape.size()) {
            return false;
        }
        for (int64_t i = 0; i < shape.size(); i++) {
            if (shape[i] != other.shape[i]) {
                return false;
            }
        }
        return thrust::equal(data.begin(), data.end(), other.data.begin());
    }

    int64_t ComputeXLength() {
        int64_t ret = 1;
        for (int i = 1; i < shape.size(); ++i) {
            ret *= shape[i];
        }
        return ret;
        }
};

template <typename DType, typename Alloc, typename FileDType>
MySimpleNDArray<DType, Alloc> LoadMySimpleNDArrayFromNumpy(const std::string &filename){
    std::vector<unsigned long> shape;
    bool fortran_order = false;
    std::vector<FileDType> data;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, data);

    thrust::host_vector<DType> myndarray_data;
    std::vector<int64_t> myndarray_shape;
    for (int64_t i = 0; i < shape.size(); i++) {
        myndarray_shape.push_back(shape[i]);
    }
    for (int64_t i = 0; i < data.size(); i++) {
        myndarray_data.push_back(data[i]);
    }
    return MySimpleNDArray<DType, Alloc>(myndarray_shape, myndarray_data);
}

//random vectorizer generator code from https://gist.github.com/ashwin/7245048
template <typename DType>
struct GenRand
{
    __device__
    DType operator () (int idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<DType> uniDist;
        randEng.discard(idx);
        return uniDist(randEng);
    }
};

template <typename DType>
MySimpleNDArray<DType, thrust::device_allocator<DType>> GenerateRandomNDArray(std::vector<int64_t> shape){
    int64_t element_num = 1;
        for (int64_t i = 0; i < shape.size(); i++) {
            element_num *= shape[i];
    }

    thrust::device_vector<DType> rVec(element_num);
    thrust::counting_iterator<int> iter_beg(0);
    thrust::counting_iterator<int> iter_end(element_num);
    thrust::transform(
        iter_beg,
        iter_end,
        rVec.begin(),
        GenRand<DType>());

    return MySimpleNDArray<DType, thrust::device_allocator<DType>>(shape, rVec);
}

