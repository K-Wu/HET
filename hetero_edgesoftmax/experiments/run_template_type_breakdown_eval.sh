nsys profile --force-overwrite true -o am_rgat.fused python -m python.RGAT.train -d am --num_layers 1 --multiply_among_weights_first_flag --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
nsys profile --force-overwrite true -o fb15k_rgat.fused python -m python.RGAT.train -d fb15k --num_layers 1 --multiply_among_weights_first_flag --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up

nsys profile --force-overwrite true -o am_rgat.unopt python -m python.RGAT.train -d am --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
nsys profile --force-overwrite true -o fb15k_rgat.unopt python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up

nsys profile --force-overwrite true -o am_rgat.compactdirect python -m python.RGAT.train -d am --num_layers 1 --compact_as_of_node_flag --compact_direct_indexing_flag --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
nsys profile --force-overwrite true -o fb15k_rgat.compactdirect python -m python.RGAT.train -d fb15k --num_layers 1 --compact_as_of_node_flag --compact_direct_indexing_flag --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up

nsys profile --force-overwrite true -o am_rgat.compactdirect.fused python -m python.RGAT.train -d am --num_layers 1 --multiply_among_weights_first_flag --compact_as_of_node_flag --compact_direct_indexing_flag --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
nsys profile --force-overwrite true -o fb15k_rgat.compactdirect.fused python -m python.RGAT.train -d fb15k --num_layers 1 --multiply_among_weights_first_flag --compact_as_of_node_flag --compact_direct_indexing_flag --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up