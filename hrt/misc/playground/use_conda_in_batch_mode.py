import subprocess

subprocess.run(
    (
        "conda run -n graphiler python3"
        " ../third_party/OthersArtifacts/graphiler/examples/HGT/HGT.py aifb"
        " 64 64"
    ),
    shell=True,
)
subprocess.run(
    (
        "conda run -n dev_dgl_torch_new python3 -m python.HGT.train -d bgs"
        " --num_layers 1 --full_graph_training --num_classes 64 --n_infeat 64"
        " -e 1"
    ),
    shell=True,
)
