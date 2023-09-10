#!/usr/bin/env python3
from .. import ref_kernels_lite
from ... import backend
from ... import utils_lite


class RefRgnnRelationalMatmulACScatterGatherListIdentical:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RgnnRelationalMatmulACScatterGatherListIdentical.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RgnnRelationalMatmulACScatterGatherListIdentical.backward
        )(*args)


class RefRgnnRelationalMatmul:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RgnnRelationalMatmul.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RgnnRelationalMatmul.backward
        )(*args)


def ref_rgnn_relational_matmul(*args):
    raise NotImplementedError(
        "ref_rgnn_relational_matmul requires Ref classes be subclasses of"
        " torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        RgnnRelationalMatmul=RefRgnnRelationalMatmul,
        debug=True,
    )(backend.rgnn_relational_matmul)(*args)


class RefRgnnRelationalMatmulNoScatterGatherList:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RgnnRelationalMatmulNoScatterGatherList.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RgnnRelationalMatmulNoScatterGatherList.backward
        )(*args)


def ref_rgnn_relational_matmul_no_scatter_gather_list(*args):
    raise NotImplementedError(
        "ref_rgnn_relational_matmul_no_scatter_gather_list requires Ref"
        " classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        RgnnRelationalMatmulNoScatterGatherList=RefRgnnRelationalMatmulNoScatterGatherList,
        debug=True,
    )(backend.rgnn_relational_matmul_no_scatter_gather_list)(*args)


class RefRgnnRelationalMatmulCompactAsOfNode:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RgnnRelationalMatmulCompactAsOfNode.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RgnnRelationalMatmulCompactAsOfNode.backward
        )(*args)


def ref_rgnn_relational_matmul_compact_as_of_node(*args):
    raise NotImplementedError(
        "ref_rgnn_relational_matmul_compact_as_of_node requires Ref classes be"
        " subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        RgnnRelationalMatmulCompactAsOfNode=RefRgnnRelationalMatmulCompactAsOfNode,
        debug=True,
    )(backend.rgnn_relational_matmul_compact_as_of_node)(*args)


class RefRgnnInnerProductNodeCompactAndNode:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RgnnInnerProductNodeCompactAndNode.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RgnnInnerProductNodeCompactAndNode.backward
        )(*args)


def ref_rgnn_inner_product_node_compact_and_node(*args):
    raise NotImplementedError(
        "ref_rgnn_inner_product_node_compact_and_node requires Ref classes be"
        " subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        RgnnInnerProductNodeCompactAndNode=RefRgnnInnerProductNodeCompactAndNode,
        debug=True,
    )(backend.rgnn_inner_product_node_compact_and_node)(*args)


class RgnnInnerProductEdgeAndNode:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RgnnInnerProductEdgeAndNode.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RgnnInnerProductEdgeAndNode.backward
        )(*args)


def ref_rgnn_inner_product_edge_and_node(*args):
    raise NotImplementedError(
        "ref_rgnn_inner_product_edge_and_node requires Ref classes be"
        " subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        RgnnInnerProductEdgeAndNode=RgnnInnerProductEdgeAndNode,
        debug=True,
    )(backend.rgnn_inner_product_edge_and_node)(*args)
