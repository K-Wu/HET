#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#include <vector>

namespace HET {
namespace OpPrototyping {
// random vectorizer generator code from https://gist.github.com/ashwin/7245048
template <typename DType>
struct GenRand {
  __device__ DType operator()(int idx) {
    thrust::default_random_engine randEng;
    thrust::uniform_real_distribution<DType> uniDist;
    randEng.discard(idx);
    return uniDist(randEng);
  }
};

template <>
struct GenRand<float4> {
  __device__ float4 operator()(int idx) {
    thrust::default_random_engine randEng;
    thrust::uniform_real_distribution<float> uniDist;
    randEng.discard(4 * idx);
    return make_float4(uniDist(randEng), uniDist(randEng), uniDist(randEng),
                       uniDist(randEng));
  }
};

// TODO: implement transpose (probably using permutation functionality if
// provided by thrust) and padding. NB: destruction where memory is reclaimed
// should be taken care of by thrust::detail::vector_base NB: pass-by-value
// arguments are deeply copied， so we should pass by reference
// TODO: switch all MySimpleNDArray argument passage to pass-by-reference
template <typename DType, typename Alloc>
class MySimpleNDArray {
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
  template <typename OtherAlloc>
  MySimpleNDArray(std::vector<int64_t> &shape,
                  thrust::detail::vector_base<DType, OtherAlloc> &data) {
    this->shape = shape;
    this->data = data;
  }
  template <typename OtherAlloc>
  MySimpleNDArray(thrust::detail::vector_base<DType, OtherAlloc> &data) {
    this->data = data;
    this->shape = {data.size()};
  }
  template <typename OtherAlloc>
  MySimpleNDArray(const MySimpleNDArray<DType, OtherAlloc> &other)
      : shape(other.shape), data(other.data) {}

  /*! \return the data pointer with type. */
  inline DType *Ptr() {
    return static_cast<DType *>(thrust::raw_pointer_cast(data.data()));
  }

  template <typename OtherAlloc>
  bool IsEqual(const MySimpleNDArray<DType, OtherAlloc> &other) const {
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

  int64_t SeastarComputeXLength() {
    int64_t ret = 1;
    for (int i = 1; i < shape.size(); ++i) {
      ret *= shape[i];
    }
    return ret;
  }

  void FillInRandomData() {
    int64_t element_num = 1;
    for (int64_t i = 0; i < shape.size(); i++) {
      element_num *= shape[i];
    }
    thrust::counting_iterator<int> iter_beg(0);
    thrust::counting_iterator<int> iter_end(element_num);
    // NB: thrust stream usage is shown here
    // https://github.com/NVIDIA/thrust/issues/1626 we will avoid the potential
    // issue of using thrust together with pytorch by only using MySimpleNDArray
    // in prototyping ops whereas in the ops exported to pytorch this class
    // MySimpleNDArray will not be used.
    thrust::transform(iter_beg, iter_end, data.begin(), GenRand<DType>());
  }
};

template <typename DType, typename Alloc, typename FileDType>
MySimpleNDArray<DType, Alloc> LoadMySimpleNDArrayFromNumpy(
    const std::string &filename) {
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

template <typename DType>
MySimpleNDArray<DType, thrust::device_allocator<DType>> GenerateRandomNDArray(
    std::vector<int64_t> shape) {
  MySimpleNDArray<DType, thrust::device_allocator<DType>> result(shape);
  result.FillInRandomData();
  return result;
}
}  // namespace OpPrototyping
}  // namespace HET