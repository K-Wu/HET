#!/usr/bin/env python3
from .. import ref_kernels_lite
from ... import backend
from ... import utils_lite


class RefRgcnLayer1SeparateCoo:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RgcnLayer1SeparateCoo.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RgcnLayer1SeparateCoo.backward
        )(*args)


def ref_rgcn_layer1_separate_coo(*args):
    raise NotImplementedError(
        "ref_rgcn_layer1_separate_coo requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite, RgcnLayer1SeparateCoo=RefRgcnLayer1SeparateCoo, debug=True
    )(backend.rgcn_layer1_separate_coo)(*args)


class RefRGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO.backward
        )(*args)


def ref_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(*args):
    raise NotImplementedError(
        "ref_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        debug=True,
        RGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO=RefRGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO,
    )(backend.rgcn_node_mean_aggregation_compact_as_of_node_separate_coo)(*args)


def ref_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo_single_sided(*args):
    raise NotImplementedError(
        "ref_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo_single_sided requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        debug=True,
        RGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO=RefRGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO,
    )(backend.rgcn_node_mean_aggregation_compact_as_of_node_separate_coo_single_sided)(
        *args
    )
