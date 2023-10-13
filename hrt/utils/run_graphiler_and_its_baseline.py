import subprocess
from .detect_pwd import (
    GRAPHILER_CONDA_ENV_NAME,
    HET_CONDA_ENV_NAME,
    create_new_results_dir,
    get_het_root_path,
    is_pwd_het_dev_root,
)

DATASETS = ["aifb", "mutag", "bgs", "am", "mag", "wikikg2", "fb15k", "biokg"]
# This is to run baseline defined in graphiler artifacts other than graphiler
BASELINE_CONDA_ENV_NAME = "dev_dgl_torch_new"


def run_grapiler(results_dir: str):
    # Run one experiment in each process to contain errors if those happen.
    for model in ["HGT", "RGAT", "RGCN"]:
        for dataset in DATASETS:
            subprocess.run(
                (
                    f"yes | conda run -n {GRAPHILER_CONDA_ENV_NAME} python3"
                    f" {get_het_root_path()}/third_party/OthersArtifacts/graphiler/examples/{model}/{model}.py"
                    f" {dataset} 64 64 >>{results_dir}/{model}.log 2>&1"
                ),
                shell=True,
            )


def run_baselines(results_dir: str):
    for model in ["HGT", "RGAT", "RGCN"]:
        for dataset in DATASETS:
            subprocess.run(
                (
                    f"yes | conda run -n {BASELINE_CONDA_ENV_NAME} python3"
                    f" {get_het_root_path()}/third_party/OthersArtifacts/graphiler/examples/{model}/{model}_baseline_standalone.py"
                    f" {dataset} 64 64"
                    f" >>{results_dir}/{model}_baseline_standalone.log 2>&1"
                ),
                shell=True,
            )


def run_seastar_RGCN(results_dir: str):
    for dataset in DATASETS:
        subprocess.run(
            (
                f"yes | conda run -n {HET_CONDA_ENV_NAME} python3 -m"
                f" python.RGCN.RGCNSingleLayer -d {dataset} --n_infeat 64"
                ' --num_classes 64 --sparse_format="csr"'
                f" >{results_dir}/seastar_rgcn.{dataset}.64.64.1.baseline.log"
                " 2>&1"
            ),
            shell=True,
        )


def run_hgl_RGCN(results_dir: str):
    # TODO: implement this
    HGL_CONDA_ENV_NAME = "hgl-env"
    raise NotImplementedError
    # output and input dimension are both d_hidden, n_heads is by default 1
    f"PYTHONPATH={get_het_root_path()}/third_party/OthersArtifacts/hgl; yes | conda run -n {HGL_CONDA_ENV_NAME} python3 {get_het_root_path()}/third_party/OthersArtifacts/hgl/test/bench_macro.py --lib=hgl --model=rgat --dataset=aifb_hetero --d_hidden=64"
    f"PYTHONPATH={get_het_root_path()}/third_party/OthersArtifacts/hgl; yes | conda run -n {HGL_CONDA_ENV_NAME} python3 {get_het_root_path()}/third_party/OthersArtifacts/hgl/test/bench_macro.py --lib=hgl --model=rgcn --dataset=aifb_hetero --d_hidden=64"


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    current_results_dir = create_new_results_dir("graphiler_")
    run_grapiler(current_results_dir)
    run_baselines(current_results_dir)
    # run_seastar_RGCN(current_results_dir)
