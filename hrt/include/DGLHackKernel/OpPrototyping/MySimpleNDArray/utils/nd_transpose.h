#pragma once
#include "flatten_coord_index.h"
#include "utils.cu.h"
#include "variadic_tricks.h"

// transposition code from
// https://github.com/OrangeOwlSolutions/cuBLAS/blob/master/Transposition.cu
#include <assert.h>
#include <conio.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <iomanip>
#include <iostream>

// convert a linear index to a linear index in the transpose
struct transpose_index_2d : public thrust::unary_function<size_t, size_t> {
  size_t m, n;

  _HOST_DEVICE_METHOD_QUALIFIER
  transpose_index_2d(size_t _m, size_t _n) : m(_m), n(_n) {}

  _HOST_DEVICE_METHOD_QUALIFIER
  size_t operator()(size_t linear_index) {
    size_t i = linear_index / n;
    size_t j = linear_index % n;

    return m * j + i;
  }
};

template <class... Ts>
struct transpose_index_nd
    : public std::tuple<typename std::remove_reference<Ts>::type...> {
  Coord<Ts...> src_coord;
  Coord<Ts...> dest_coord;
  decay_args_tuple<decltype(a.shape)>::type
      permutation;  // permutation(src_coord) == dest_coord
  _HOST_DEVICE_METHOD_QUALIFIER
  transpose_index_nd(Coord<Ts...> src_coord, Coord<Ts...> dest_coord,
                     decay_args_tuple<decltype(a.shape)>::type permutation)
      : src_coord(src_coord),
        dest_coord(dest_coord),
        permutation(permutation) {}

  _HOST_DEVICE_METHOD_QUALIFIER
  size_t operator()(size_t linear_index) {
    auto coord = convert_flat_index_to_tuple(linear_index, src_coord.shape);
    auto permuted_coord = permute_coord(coord, permutation);
    return get_flattened_index(permuted_coord, dest_coord.shape);
  }
};

// transpose an M-by-N array
template <typename T>
void transpose_2d(size_t m, size_t n, thrust::device_vector<T>& src,
                  thrust::device_vector<T>& dst) {
  thrust::counting_iterator<size_t> indices(0);

  thrust::gather(
      thrust::make_transform_iterator(indices, transpose_index_2d(n, m)),
      thrust::make_transform_iterator(indices, transpose_index_2d(n, m)) +
          dst.size(),
      src.begin(), dst.begin());
}

template <typename T, typename CoordType, typename PermutationType>
void transpose_nd(CoordType src_coord, CoordType dest_coord,
                  PermutationType permutation, thrust::device_vector<T>& src,
                  thrust::device_vector<T>& dst) {
  thrust::counting_iterator<size_t> indices(0);

  thrust::gather(
      thrust::make_transform_iterator(
          indices, transpose_index_nd(src_coord, dest_coord, permutation)),
      thrust::make_transform_iterator(
          indices, transpose_index_nd(src_coord, dest_coord, permutation)) +
          dst.size(),
      src.begin(), dst.begin());
}

// print an M-by-N array
template <typename T>
void print_2d(size_t m, size_t n, thrust::device_vector<T>& d_data) {
  thrust::host_vector<T> h_data = d_data;

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++)
      std::cout << std::setw(8) << h_data[i * n + j] << " ";
    std::cout << "\n";
  }
}

void test() {
  size_t m = 5;  // number of rows
  size_t n = 4;  // number of columns

  // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
  thrust::device_vector<double> data(m * n, 1.);
  data[1] = 2.;
  data[3] = 3.;

  std::cout << "Initial array" << std::endl;
  print_2d(m, n, data);

  std::cout << "Transpose 2d array - Thrust" << std::endl;
  thrust::device_vector<double> transposed_thrust(m * n);
  transpose_2d(m, n, data, transposed_thrust);
  print_2d(n, m, transposed_thrust);

  std::cout << "Transpose 2d array - cuBLAS" << std::endl;
  thrust::device_vector<double> transposed_cuBLAS(m * n);
  double* dv_ptr_in = thrust::raw_pointer_cast(data.data());
  double* dv_ptr_out = thrust::raw_pointer_cast(transposed_cuBLAS.data());
  double alpha = 1.;
  double beta = 0.;
  cublasHandle_t handle;
  cuda_err_chk(cublasCreate(&handle));
  cuda_err_chk(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha,
                           dv_ptr_in, n, &beta, dv_ptr_in, n, dv_ptr_out, m));
  print_2d(n, m, transposed_cuBLAS);

  getch();

  return 0;
}