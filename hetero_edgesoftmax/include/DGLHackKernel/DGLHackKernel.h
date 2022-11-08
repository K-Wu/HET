#pragma once
// NB: the order here is sensitive. To suppress clang-format change the order, I
// deliberately add new line among these include statements for the time being.

#include <optional>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <npy.hpp>
#include "MySimpleNDArray/MySimpleNDArray.h"

#include "../MyHyb/MyHyb.h"
#include "../hetero_edgesoftmax.h"
#include "../utils.cu.h"

#include "DGLHackUtils.h"
#include "GAT/FusedGAT.cu.h"
#include "GAT/FusedGATBackward.cu.h"
#include "OpPrototyping/GATProtoOps.h"
#include "OpPrototyping/HGTProtoOps.h"
#include "OpPrototyping/RGCNProtoOps.h"
#include "RGAT/RGATLayersKernels.cu.h"
#include "RGCN/RGCNLayer1BackwardMyHYB.h"
#include "RGCN/RGCNLayer1MyHYB.h"
#include "RGCN/RGCNLayersBackwardKernels.cu.h"
#include "RGCN/RGCNLayersBackwardKernelsCOO.cu.h"
#include "RGCN/RGCNLayersKernels.cu.h"
#include "RGCN/RGCNLayersKernelsCOO.cu.h"

#include "HGT/HGTExperimental.h"
#include "HGT/HGTLayersBackwardKernels.cu.h"

#include "DGLHackKernelInit.h"