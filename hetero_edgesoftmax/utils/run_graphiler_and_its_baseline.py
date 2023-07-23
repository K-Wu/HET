import subprocess
from .detect_pwd import (
    GRAPHILER_CONDA_ENV_NAME,
    HET_CONDA_ENV_NAME,
    create_new_results_dir,
    get_het_root_path,
)


def run_grapiler(results_dir: str):
    for model in ["HGT", "RGAT", "RGCN"]:
        subprocess.run(
            f"yes | conda run -n {GRAPHILER_CONDA_ENV_NAME} python3 {get_het_root_path()}/third_party/OthersArtifacts/graphiler/examples/{model}/{model}.py all 64 64 >{results_dir}/{model}.log 2>&1",
            shell=True,
        )


def run_baselines(results_dir: str):
    for model in ["HGT", "RGAT", "RGCN"]:
        subprocess.run(
            f"yes | conda run -n {HET_CONDA_ENV_NAME} python3 {get_het_root_path()}/third_party/OthersArtifacts/graphiler/examples/{model}/{model}_baseline_standalone.py all 64 64 >{results_dir}/{model}_baseline_standalone.log 2>&1",
            shell=True,
        )


if __name__ == "__main__":
    current_results_dir = create_new_results_dir("graphiler_")
    run_grapiler(current_results_dir)
    run_baselines(current_results_dir)
