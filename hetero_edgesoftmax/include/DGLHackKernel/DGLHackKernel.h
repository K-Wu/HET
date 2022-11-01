#pragma once
// NB: the order here is sensitive. To suppress clang-format change the order, I
// deliberately add new line among these include statements for the time being.

#include <optional>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <npy.hpp>
#include "MySimpleNDArray.h"

#include "../MyHyb/MyHyb.h"
#include "../hetero_edgesoftmax.h"
#include "../utils.cu.h"

#include "DGLHackUtils.h"
#include "FusedGAT.cu.h"
#include "FusedGATBackward.cu.h"
#include "OpPrototyping/GATOps.h"
#include "OpPrototyping/HGTOps.h"
#include "OpPrototyping/RGCNOps.h"
#include "RGCNLayer1BackwardMyHYB.h"
#include "RGCNLayer1MyHYB.h"
#include "RGCNLayersBackwardKernels.cu.h"
#include "RGCNLayersBackwardKernelsCOO.cu.h"
#include "RGCNLayersKernels.cu.h"
#include "RGCNLayersKernelsCOO.cu.h"

#include "HGTExperimental.h"
#include "HGTLayersBackwardKernels.cu.h"

#include "DGLHackKernelInit.h"