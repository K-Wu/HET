#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/FusedGAT.cu.h"
#include "DGLHackKernel/FusedGATBackward.cu.h"
#include "DGLHackKernel/RGATLayersBackwardKernels.cu.h"
#include "DGLHackKernel/RGATLayersKernels.cu.h"

void RelationalFusedGATKernel_wrapper_integratedcsr(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, double slope) {}

void BackwardRelationalFusedGATKernel_wrapper_integratedcsr(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes, at::Tensor& feat_src,
    at::Tensor& el, at::Tensor& er, at::Tensor& sum, at::Tensor& exp,
    at::Tensor& ret, at::Tensor& gradout, at::Tensor& grad_feat_src,
    at::Tensor& grad_el, at::Tensor& grad_er, double slope) {}

void RGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weights, at::Tensor& input,
    at::Tensor& ret) {}

void BackwardRGATRelationalMatMul_wrapper_separatecoo(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_node_indices,
    at::Tensor& separate_coo_eids, at::Tensor& weights_transposed,
    at::Tensor& input, at::Tensor& gradout, at::Tensor& grad_input,
    at::Tensor& grad_weights) {}

void BackwardRGATRelationalFusedGATKernelCompactAsOfNode_wrapper_integratedcsr(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& feat_compact, at::Tensor& el_compact,
    at::Tensor& er_compact, at::Tensor& sum, at::Tensor& exp, at::Tensor& ret,
    at::Tensor& gradout, at::Tensor& grad_feat_compact,
    at::Tensor& grad_el_compact, at::Tensor& grad_er_compact, double slope) {}
void RGATRelationalFusedGATKernelCompactAsOfNode_wrapper_integratedcsr(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& feat_comapct, at::Tensor& el_compact,
    at::Tensor& er_compact, at::Tensor& sum, at::Tensor& exp, at::Tensor& ret,
    double slope) {}
void BackwardRGATRelationalMatMulCompactAsOfNode_wrapper_unique_rel_node_indices(
    at::Tensor& unqiue_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx, at::Tensor& weight,
    at::Tensor& node_feat, at::Tensor& ret, at::Tensor& gradout,
    at::Tensor& grad_weight, at::Tensor& grad_node_feat) {}
void RGATRelationalMatMulCompactAsOfNode_wrapper_unique_rel_node_indices(
    at::Tensor& unique_srcs_and_dests_rel_ptr,
    at::Tensor& unique_srcs_and_dests_node_idx, at::Tensor& weight,
    at::Tensor& node_feat, at::Tensor& ret) {}