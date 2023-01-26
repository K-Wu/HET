#!/usr/bin/env python3
from .. import ref_kernels_lite
from ... import backend
from ... import utils_lite


class RefRelationalFusedGatSeparateCOO:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RelationalFusedGatSeparateCOO.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RelationalFusedGatSeparateCOO.backward
        )(*args)


class RefRelationalFusedGatCompactAsOfNodeSeparateCOODualUniqueNodeList:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RelationalFusedGatCompactAsOfNodeSeparateCOODualUniqueNodeList.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RelationalFusedGatCompactAsOfNodeSeparateCOODualUniqueNodeList.backward
        )(*args)


class RefRelationalFusedGatCompactAsOfNodeSeparateCOO:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RelationalFusedGatCompactAsOfNodeSeparateCOO.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RelationalFusedGatCompactAsOfNodeSeparateCOO.backward
        )(*args)


class RefRelationalFusedGatCSR:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RelationalFusedGatCSR.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RelationalFusedGatCSR.backward
        )(*args)


def ref_relational_fused_gat_csr(*args):
    raise NotImplementedError(
        "ref_relational_fused_gat_csr requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite, debug=True, RelationalFusedGatCSR=RefRelationalFusedGatCSR
    )(backend.relational_fused_gat_csr)(*args)


class RefRgatRelationalFusedGATCompactAsOfNodeCSR:
    @staticmethod
    def forward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite, debug=True)(
            backend.RgatRelationalFusedGATCompactAsOfNodeCSR.forward
        )(*args)

    @staticmethod
    def backward(*args):
        return utils_lite.reroute_namespace(K=ref_kernels_lite)(
            backend.RgatRelationalFusedGATCompactAsOfNodeCSR.backward
        )(*args)


def ref_relational_fused_gat_compact_as_of_node_separate_coo_dual_unique_node_list(
    *args,
):
    raise NotImplementedError(
        "relational_fused_gat_compact_as_of_node_separate_coo_dual_unique_node_list requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        debug=True,
        RelationalFusedGatCompactAsOfNodeSeparateCOODualUniqueNodeList=RefRelationalFusedGatCompactAsOfNodeSeparateCOODualUniqueNodeList,
    )(
        backend.relational_fused_gat_compact_as_of_node_separate_coo_dual_unique_node_list
    )(
        *args
    )


def ref_relational_fused_gat_separate_coo(*args):
    raise NotImplementedError(
        "relational_fused_gat_separate_coo requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        debug=True,
        RelationalFusedGatSeparateCOO=RefRelationalFusedGatSeparateCOO,
    )(backend.relational_fused_gat_separate_coo)(*args)


def ref_relational_fused_gat_compact_as_of_node(*args):
    raise NotImplementedError(
        "ref_relational_fused_gat_compact_as_of_node requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        debug=True,
        RgatRelationalFusedGATCompactAsOfNodeCSR=RefRgatRelationalFusedGATCompactAsOfNodeCSR,
    )(backend.relational_fused_gat_compact_as_of_node)(*args)


def ref_relational_fused_gat_compact_as_of_node_separate_coo(*args):
    raise NotImplementedError(
        "ref_relational_fused_gat_compact_as_of_node_separate_coo requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        debug=True,
        RelationalFusedGatCompactAsOfNodeSeparateCOO=RefRelationalFusedGatCompactAsOfNodeSeparateCOO,
    )(backend.relational_fused_gat_compact_as_of_node_separate_coo)(*args)


def ref_relational_fused_gat_compact_as_of_node_separate_coo_single_sided(*args):
    raise NotImplementedError(
        "ref_relational_fused_gat_compact_as_of_node_separate_coo_single_sided requires Ref classes be subclasses of torch.autograd.Function"
    )
    return utils_lite.reroute_namespace(
        K=ref_kernels_lite,
        debug=True,
        RelationalFusedGatCompactAsOfNodeSeparateCOODualUniqueNodeList=RefRelationalFusedGatCompactAsOfNodeSeparateCOODualUniqueNodeList,
    )(backend.relational_fused_gat_compact_as_of_node_separate_coo_single_sided)(*args)
