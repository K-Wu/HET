#!/usr/bin/env bash
cd ./build/exe/src
cuobjdump ./test_DGLHackKernel.cu.exe -xelf all
# The prompt will print something like
# Extraccting ELF file    1: xxx.sm_xx.cubin
nvdisasm -gi test_DGLHackKernel.sm_70.cubin >dump.log
# We can alternatively do cuobjdump -sass ./test_DGLHackKernel.cu.exe to get code parsible by https://github.com/cloudcores/CuAssembler/