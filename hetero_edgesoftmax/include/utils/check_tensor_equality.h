#pragma once
#include "nd_transpose.h"
// TODO: we are to delivery equality check of two tensors with some coordinate
// permutation, i.e., n-dimensional transpose
// TODO: based on that, we can invoke referential spmm, sddmm implementation
// from e.g. sputnik to check the correctness of our routines