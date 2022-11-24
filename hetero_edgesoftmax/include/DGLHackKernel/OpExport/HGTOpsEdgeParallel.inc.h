#pragma once
#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"
#include "EdgeSoftmax_1/EdgeSoftmaxCSR.h"

namespace HET {
namespace TorchExport {
namespace HGT {
namespace FwProp {
namespace IntegratedCSR {
namespace EdgeParallel {
void full_graph_message_mean_aggregation() {
  // We may use HGTTriviallyEdgeParallelCompactAsOfNodeNodeMeanAggregation in
  // hetero_edgesoftmax/include/DGLHackKernel/HGT/HGTLayersForwardKernels.cu.h
  assert(0 && "Not implemented yet");
}
void full_graph_edge_softmax_ops() { assert(0 && "Not implemented yet"); }
}  // namespace EdgeParallel
}  // namespace IntegratedCSR
}  // namespace FwProp
namespace BckProp {
namespace IntegratedCSR {
namespace EdgeParallel {
void full_graph_message_mean_aggregation() {
  assert(0 && "Not implemented yet");
}
void full_graph_edge_softmax_ops() { assert(0 && "Not implemented yet"); }
}  // namespace EdgeParallel
}  // namespace IntegratedCSR
}  // namespace BckProp
}  // namespace HGT
}  // namespace TorchExport
}  // namespace HET
