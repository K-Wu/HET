# HET

The HET Hetero-GNN Kernel Optimization and Code Generation project.

## Dependencies
The following repos, as submodules in `third_party/`, are required. Please recursively clone these submodules.

[CUTLASS](https://github.com/NVIDIA/cutlass)

[sputnik](https://github.com/google-research/sputnik)

Besides, as we register our kernels in PyTorch (optional cmake build component), pytorch-dev and libtorch are also required.

## Code Quality
[![CodeFactor](https://www.codefactor.io/repository/github/k-wu/hetero_edgesoftmax/badge?s=34a94a8b3a8b3d83b6582edc6e24b1e5d0a207b9)](https://www.codefactor.io/repository/github/k-wu/hetero_edgesoftmax)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9c41863c914e4153883f24eeff256280)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=K-Wu/hetero_edgesoftmax&amp;utm_campaign=Badge_Grade)

## Warning on Infidel Sort by Src Out Degree and by Etype
This repo contains data sorted in infidel manner (in [[hetero_edgesoftmax/data]]) with ".infidel_sorted" mark and utility to do such sort (in [[hetero_edgesoftmax/python/utils/coo_sorters.py]]). This is a sort mechanism solely for load balance in RGCN and is not a general purpose sort. The sorted elements, i.e., source node index or etype, are reindexed while other elements, i.e., one of source node index or etype, and eids and destination node index, are not. Also "transposed.<dataset_name>.coo.infidel_sorted.by_srcs_outgoing_freq.<element_name>.npy" are sorted after transposed so using the same eid should refer to the same edge in the original data.

## CMake Commands
We need to set BUILD_TEST and BUILD_BENCHMARK as they will be passed on to sub-repo sputnik and build necessary components.
```
mkdir build
cd build
cmake .. -DCUDA_ARCHS="70;75" -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DBUILD_GENERATED=ON -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DBUILD_TORCH_API=ON
```

for debugging purpose, you may invoke the following command after cmake configuration.
```
cmake --build . --verbose
```

## Contributing in submodules
To change the url of a submodule, use the following command:
```
git submodule set-url </reporoot/to/submodule_name> <new_url>
```

To set a different push url than the fetch url, inside the submodule directory, use the following command:
```
git remote set-url --push <branch_name> <new_url>
```

Try the following command to push in a submodule:
```
git push origin HEAD:master
```
