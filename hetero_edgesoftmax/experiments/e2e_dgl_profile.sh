#!/bin/bash -x
cd ~/dgl/examples/pytorch/hgt # the nvtx-annotated repo could be retrieved at https://github.com/K-Wu/dgl-nvtx
nsys profile --force-overwrite true -o hgt_acm python train_acm.py --n_epoch 1
