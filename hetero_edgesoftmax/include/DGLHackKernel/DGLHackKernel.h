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
#include "../utils.h"

#include "DGLHackUtils.h"
#include "FusedGAT.h"
#include "FusedGATBackward.h"
#include "RGCNLayer1BackwardMyHYB.h"
#include "RGCNLayer1MyHYB.h"
#include "RGCNLayers.h"
#include "RGCNLayersBackward.h"

#include "HGTBackPropGradientSmAFusion.h"
#include "HGTExperimental.h"

#include "DGLHackKernelInit.h"