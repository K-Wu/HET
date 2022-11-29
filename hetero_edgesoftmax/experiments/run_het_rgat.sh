#!/bin/bash -x
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --n_head 8 --n_infeat 128
python -m python.RGAT.train -d wikikg2 --num_layers 1 --full_graph_training --num_classes 8 --n_head 8 --n_infeat 128
python -m python.RGAT.train -d ogbn-mag --num_layers 1 --full_graph_training --num_classes 8 --n_head 8 --n_infeat 128
