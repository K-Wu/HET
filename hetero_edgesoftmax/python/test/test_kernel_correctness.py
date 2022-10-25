from .. import kernels as K
from .. import utils
import networkx as nx
import torch


def test_correctness_transpose_csr():
    # TODO: recognize and adhere to torch current stream in our C++ CUDA kernels
    g = utils.generate_synthetic_graph(100, 0.1, seed=0)
    S = nx.to_scipy_sparse_matrix(g, format="csr")

    original_row_ptrs = torch.from_numpy(S.indptr).long()
    original_col_idxes = torch.from_numpy(S.indices).long()
    # generate random eids and etypes
    original_eids = torch.randint(0, 100, (original_col_idxes.size(0),))
    original_reltypes = torch.randint(0, 10, (original_col_idxes.size(0),))

    (
        transposed_row_ptrs,
        transposed_col_idxes,
        transposed_eids,
        transposed_reltypes,
    ) = K.transpose_csr(
        original_row_ptrs, original_col_idxes, original_eids, original_reltypes
    )

    (
        double_transposed_row_ptrs,
        double_transposed_col_idxes,
        double_transposed_eids,
        double_transposed_reltypes,
    ) = K.transpose_csr(
        transposed_row_ptrs, transposed_col_idxes, transposed_eids, transposed_reltypes
    )
    torch.cuda.synchronize()
    assert torch.equal(original_row_ptrs, double_transposed_row_ptrs)
