#pragma once
// NB: this should be headers to expose APIs to outside the project, and we
// need to avoid the reference of this header in this project.

// NB: the order here is sensitive. To suppress clang-format change the order, I
// deliberately add new line among these include statements for the time being.

#include <optional>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <npy.hpp>
#include "OpPrototyping/MySimpleNDArray/MySimpleNDArray.h"

#include "../MyHyb/MyHyb.h"
#include "../hetero_edgesoftmax.h"
#include "../utils.cu.h"

#include "GAT/FusedGAT.cu.h"
#include "GAT/FusedGATBackward.cu.h"
#include "OpPrototyping/GATProtoOps.h"
#include "OpPrototyping/HGTProtoOps.h"
#include "OpPrototyping/RGCNProtoOps.h"
#include "RGAT/RGATKernelsSeparateCOO.cu.h"
#include "RGAT/RGATKernelsSeparateCSR.cu.h"
#include "RGCN/RGCNLayer1BackwardMyHYB.h"
#include "RGCN/RGCNLayer1MyHYB.h"
#include "RGCN/SeastarRGCNBackwardKernels.cu.h"
#include "RGCN/SeastarRGCNKernels.cu.h"

#include "HGT/HGTBackwardKernels.cu.h"
#include "HGT/HGTForwardExperimental.cu.h"

#include "DGLHackKernelInit.h"
