# Use cuda_gpu_kern_sum and aggregate to get Hector kernels consumed time and PyTorch kernels consumed time, and then substract it from the total time to get others' time
# Alternatively, we can still use cuda_gpu_kern_sum to get Hecto kernels consumed time and PyTorch kernels consumed time. But we count all the kernels time in the training range and substract it from the range duration to get others' time
nsys profile --force-overwrite true -o mutag_rgat.bg.breakdown python -m python.RGAT.train -d mutag --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
nsys profile --force-overwrite true -o fb15k_rgat.bg.breakdown python -m python.RGAT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up

nsys profile --force-overwrite true -o mutag_hgt.bg.breakdown python -m python.HGT.train -d mutag --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up
nsys profile --force-overwrite true -o fb15k_hgt.bg.breakdown python -m python.HGT.train -d fb15k --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64 -e 1 --no_warm_up