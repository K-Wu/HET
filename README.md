# HET [![CodeFactor](https://www.codefactor.io/repository/github/k-wu/het/badge?s=7c0ed599f222a217c8682e478c593e0ca4b434da)](https://www.codefactor.io/repository/github/k-wu/het) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/9c41863c914e4153883f24eeff256280)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=K-Wu/hetero_edgesoftmax&amp;utm_campaign=Badge_Grade) [![DeepSource](https://app.deepsource.com/gh/K-Wu/HET.svg/?label=active+issues&show_trend=true&token=hA6-EP3JaW1Y08vzTD9kDiAr)](https://app.deepsource.com/gh/K-Wu/HET/?ref=repository-badge)

The HET Hetero-GNN Kernel Optimization and Code Generation project.

## Dependencies
The following repos, as submodules in `third_party/`, are required. Please recursively clone these submodules.

[CUTLASS](https://github.com/NVIDIA/cutlass)

[sputnik](https://github.com/google-research/sputnik)

Besides, as we register our kernels in PyTorch (optional cmake build component), pytorch-dev and libtorch are also required.

## CMake commands
We need to set BUILD_TEST and BUILD_BENCHMARK as they will be passed on to sub-repo sputnik and build necessary components.
```
mkdir build
cd build
cmake .. -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DBUILD_GENERATED=ON -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DBUILD_TORCH_API=ON
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

## What's in a name?
Here is the reference for the naming of `hrt` and `pyctor`.

* [Pictor - Wikipedia](https://en.wikipedia.org/wiki/Pictor)
* [Heartland (United States) - Wikipedia](https://en.wikipedia.org/wiki/Heartland_(United_States))

## Contact
Kun Wu kunwu2 (at) illinois (dot) edu  [![wakatime](https://wakatime.com/badge/github/K-Wu/hetero_edgesoftmax.svg)](https://wakatime.com/badge/github/K-Wu/hetero_edgesoftmax)