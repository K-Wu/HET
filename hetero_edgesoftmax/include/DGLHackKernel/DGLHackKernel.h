#pragma once
#include <optional>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include "MySimpleNDArray.h"
#include "../utils.h"
#include "../MyHyb/MyHyb.h"
#include "../hetero_edgesoftmax.h"
#include "DGLHackUtils.h"
#include "FusedGAT.h"
#include "FusedGATBackward.h"

#include "RGCNLayer1.h"
#include "RGCNLayer1Backward.h"
#include "RGCNLayer1MyHYB.h"
#include "RGCNLayer1BackwardMyHYB.h"
#include "DGLHackKernelInit.h"