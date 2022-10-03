# hetero_edgesoftmax

## Dependencies
The following repos, as submodules in `third_party/`, are required. Please recursively clone these submodules.

[CUTLASS](https://github.com/NVIDIA/cutlass)
[sputnik](https://github.com/google-research/sputnik)

## Code Quality
[![CodeFactor](https://www.codefactor.io/repository/github/k-wu/hetero_edgesoftmax/badge?s=34a94a8b3a8b3d83b6582edc6e24b1e5d0a207b9)](https://www.codefactor.io/repository/github/k-wu/hetero_edgesoftmax)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9c41863c914e4153883f24eeff256280)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=K-Wu/hetero_edgesoftmax&amp;utm_campaign=Badge_Grade)

## CMake Commands
We need to set BUILD_TEST and BUILD_BENCHMARK as they will be passed on to sub-repo sputnik and build necessary components.
```
mkdir build
cd build
cmake .. -DCUDA_ARCHS="70;75" -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DBUILD_GENERATED=ON
```

for debugging purpose, you may invoke the following command after cmake configuration.
```
cmake --build . --verbose
```