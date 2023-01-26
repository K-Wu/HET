#!/usr/bin/env python3
from .. import ref_kernels_lite
from ... import backend
from ... import utils_lite


class RefHGTFullGraphHeteroAttentionOps:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.HGTFullGraphHeteroAttentionOps.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.HGTFullGraphHeteroAttentionOps.backward
        )(*args)


def ref_hgt_full_graph_hetero_attention_ops_coo(*args):
    raise NotImplementedError(
        "ref_hgt_full_graph_hetero_attention_ops_cooc requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        HGTFullGraphHeteroAttentionOps=RefHGTFullGraphHeteroAttentionOps,
        debug=True,
    )(backend.hgt_full_graph_hetero_attention_ops_cooc)(*args)


class RefHGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCSR:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.HGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCSR.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.HGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCSR.backward
        )(*args)


def ref_hgt_full_graph_message_calc_edge_softmax_and_message_mean_aggregation_csr(
    *args,
):
    raise NotImplementedError(
        "ref_hgt_full_graph_message_calc_edge_softmax_and_message_mean_aggregation_csr requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        HGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCSR=RefHGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCSR,
        debug=True,
    )(
        backend.hgt_full_graph_message_calc_edge_softmax_and_message_mean_aggregation_csr
    )(
        *args
    )


class RefHGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.HGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.HGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR.backward
        )(*args)


def ref_hgt_full_graph_edge_softmax_and_message_mean_aggregation_csr(*args):
    raise NotImplementedError(
        "ref_hgt_full_graph_edge_softmax_and_message_mean_aggregation_csr requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        HGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR=RefHGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR,
        debug=True,
    )(backend.hgt_full_graph_edge_softmax_and_message_mean_aggregation_csr)(*args)
