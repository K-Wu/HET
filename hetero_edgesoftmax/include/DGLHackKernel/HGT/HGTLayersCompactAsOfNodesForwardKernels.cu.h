#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/HGTPreprocessing.h"
#include "EdgeAttention_4/mysgemm_functor.cu.h"

// This is to calculate the product of (sW) and t where (sW) is stored per edge
// and t is stored per node.
constexpr auto HGTCompactAsOfNodesEdgeAttentionSecondStage =
    GeneralEdgeMessageMultiplyNodeFeature<float, true, false, float **>;