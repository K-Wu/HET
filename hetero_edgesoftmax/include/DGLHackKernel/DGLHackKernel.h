#pragma once
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <memory>
#include <npy.hpp>
#include <optional>
#include "../MyHyb/MyHyb.h"
#include "../hetero_edgesoftmax.h"
#include "../utils.h"
#include "DGLHackUtils.h"
#include "FusedGAT.h"
#include "FusedGATBackward.h"
#include "MySimpleNDArray.h"

#include "DGLHackKernelInit.h"
#include "HGTBackPropGradientSmAFusion.h"
#include "HGTExperimental.h"
#include "RGCNLayer1BackwardMyHYB.h"
#include "RGCNLayer1MyHYB.h"
#include "RGCNLayers.h"
#include "RGCNLayersBackward.h"
