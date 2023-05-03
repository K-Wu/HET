// from
// https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
// kernels defined in this file are simply wrapped up in
// hetero_edgesoftmax/python/kernels to provide python API, then used to
// define autograd functions and layers in hetero_edgesoftmax/python/kernels,
// which is finally referred to by end2end cases in
// hetero_edgesoftmax/python/<model name>/.*.py
// NB: This contains wrapper versions for python api export originally
// implemented at
// [[hetero_edgesoftmax/include/DGLHackKernel/RGCN/SeastarRGCNKernels.cu.h]].
// Please update accordingly whenever there is update.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <iostream>
#include <map>

#include "DGLHackKernel/DGLHackKernel.h"
#include "hetero_edgesoftmax.h"

#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"

// NB: torch builtin by default uses int64_t and single-precision float
#include "DGLHackKernel/OpExport/DataConverters.inc.h"
#include "DGLHackKernel/OpExport/GATOps.inc.h"
#include "DGLHackKernel/OpExport/HGTOps.inc.h"
#include "DGLHackKernel/OpExport/HGTOpsEdgeParallel.inc.h"
#include "DGLHackKernel/OpExport/RGATOps.inc.h"
#include "DGLHackKernel/OpExport/RGCNOps.inc.h"
#include "DGLHackKernel/OpExport/RGNNOps.inc.h"
#include "DGLHackKernel/OpExport/UtilityAndPlayground.inc.h"
#include "generated/DebugInfo.inc.h"

// NB: let's establish a convention of namespace hierarchy, i.e.,
// HET::TorchExport::ModelName::FwOrBwProp::FullGraphOrMinibatch::FormatSpecificationsEGIntegratedCSR::ComputeScheduleSpecificationsEGEdgeParallel::KernelName_VariantName
// In specific case where there is one level/field of namespace missing, e.g.,
// FullGraphOrMinibatch, we can just skip and move to the next inner level.
