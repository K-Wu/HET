cd ./build/exe/src
cuobjdump ./test_DGLHackKernel.cu.exe -xelf all
nvdisasm -gi test_DGLHackKernel.sm_70.cubin >dump.log