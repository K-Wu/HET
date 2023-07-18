The op.cu.cc ``include'' all the .inc.h files, either in the code directory, or the generated subdirectory in the build dir.
In this way, the build process could produce a shared library as a torch extension.

The original op.cu.cc stored in this directory is superceded by generated op.cu.cc in build dir.
The generation logic is in hetero_edgesoftmax/buildutils/genutils

from https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake kernels defined in this file are simply wrapped up in hetero_edgesoftmax/python/kernels to provide python API, then used to define autograd functions and layers in hetero_edgesoftmax/python/kernels, which is finally referred to by end2end cases in hetero_edgesoftmax/python/<model name>/.*.py

NB: This contains wrapper versions for python api export originally implemented at [[hetero_edgesoftmax/include/DGLHackKernel/RGCN/SeastarRGCNKernels.cu.h]]. Please update accordingly whenever there is update.

NB: torch builtin by default uses int64_t and single-precision floatNB: let's establish a convention of namespace hierarchy, i.e., HET::TorchExport::ModelName::FwOrBckProp::FullGraphOrMinibatch::FormatSpecificationsEGIntegratedCSR::ComputeScheduleSpecificationsEGEdgeParallel::KernelName_VariantName
In specific case where there is one level/field of namespace missing, e.g., FullGraphOrMinibatch, we can just skip and move to the next inner level.