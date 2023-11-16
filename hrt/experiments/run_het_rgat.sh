#!/bin/bash -x
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128
python -m python.RGAT.train -d wikikg2 --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128
python -m python.RGAT.train -d mag --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128

python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --sort_by_src
python -m python.RGAT.train -d wikikg2 --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --sort_by_src
python -m python.RGAT.train -d mag --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --sort_by_src

python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --compact_as_of_node_flag
python -m python.RGAT.train -d wikikg2 --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --compact_as_of_node_flag
python -m python.RGAT.train -d mag --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --compact_as_of_node_flag

python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --sort_by_src --compact_as_of_node_flag
python -m python.RGAT.train -d wikikg2 --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --sort_by_src --compact_as_of_node_flag
python -m python.RGAT.train -d mag --num_layers 1 --full_graph_training --num_classes 8 --num_heads 8 --n_infeat 128 --sort_by_src --compact_as_of_node_flag

python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128 --multiply_among_weights_first_flag
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 8 --n_infeat 128 --compact_as_of_node_flag --multiply_among_weights_first_flag
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --multiply_among_weights_first_flag
python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 --compact_as_of_node_flag --multiply_among_weights_first_flag