# source this script
conda activate dev_dgl_torch_annotated
#nsys profile --force-overwrite true -o mutag_rgat_showcase_a python ../third_party/OthersArtifacts/graphiler/examples/RGAT/RGAT_baseline_standalone.py mutag 64 64 a+
nsys profile --force-overwrite true -o mutag_rgat_showcase_b python ../third_party/OthersArtifacts/graphiler/examples/RGAT/RGAT_baseline_standalone.py mutag 64 64 b+

#nsys profile --force-overwrite true -o fb15k_rgat_showcase_a python ../third_party/OthersArtifacts/graphiler/examples/RGAT/RGAT_baseline_standalone.py fb15k 64 64 a+
nsys profile --force-overwrite true -o fb15k_rgat_showcase_b python ../third_party/OthersArtifacts/graphiler/examples/RGAT/RGAT_baseline_standalone.py fb15k 64 64 b-

#nsys profile --force-overwrite true -o mutag_hgt_showcase_a python ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT_baseline_standalone.py mutag 64 64 a+
#nsys profile --force-overwrite true -o mutag_hgt_showcase_b python ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT_baseline_standalone.py mutag 64 64 b+
nsys profile --force-overwrite true -o mutag_hgt_showcase_c python ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT_baseline_standalone.py mutag 64 64 c+
#nsys profile --force-overwrite true -o mutag_hgt_showcase_d python ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT_baseline_standalone.py mutag 64 64 d+

#nsys profile --force-overwrite true -o fb15k_hgt_showcase_a python ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT_baseline_standalone.py fb15k 64 64 a+
#nsys profile --force-overwrite true -o fb15k_hgt_showcase_b python ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT_baseline_standalone.py fb15k 64 64 b+
nsys profile --force-overwrite true -o fb15k_hgt_showcase_c python ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT_baseline_standalone.py fb15k 64 64 c+
#nsys profile --force-overwrite true -o fb15k_hgt_showcase_d python ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT_baseline_standalone.py fb15k 64 64 d+

nsys profile --force-overwrite true -o fb15k_rgcn_showcase_c python ../third_party/OthersArtifacts/graphiler/examples/RGCN/RGCN_baseline_standalone.py fb15k 64 64 c+
nsys profile --force-overwrite true -o mutag_rgcn_showcase_c python ../third_party/OthersArtifacts/graphiler/examples/RGCN/RGCN_baseline_standalone.py mutag 64 64 c+