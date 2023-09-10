#!/bin/bash -x
python -m python.RGAT.train -d bgs --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128 >bgs_128_8.out 2>&1
python -m python.RGAT.train -d bgs --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128 --multiply_among_weights_first_flag >bgs_128_8_mul_am_weights.out 2>&1
python -m python.RGAT.train -d bgs --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128 --compact_as_of_node_flag --multiply_among_weights_first_flag >bgs_128_8_compact_mul_am_weights.out 2>&1
python -m python.RGAT.train -d bgs --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 >bgs_64_64.out 2>&1
python -m python.RGAT.train -d bgs --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --multiply_among_weights_first_flag >bgs_64_64_mul_am_weights.out 2>&1
python -m python.RGAT.train -d bgs --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --compact_as_of_node_flag --multiply_among_weights_first_flag >bgs_64_64_compact_mul_am_weights.out 2>&1

python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128 >fb15k_128_8.out 2>&1
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128 --multiply_among_weights_first_flag >fb15k_128_8_mul_am_weights.out 2>&1
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128 --compact_as_of_node_flag --multiply_among_weights_first_flag >fb15k_128_8_compact_mul_am_weights.out 2>&1
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 >fb15k_64_64.out 2>&1
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --multiply_among_weights_first_flag >fb15k_64_64_mul_am_weights.out 2>&1
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --compact_as_of_node_flag --multiply_among_weights_first_flag >fb15k_64_64_compact_mul_am_weights.out 2>&1