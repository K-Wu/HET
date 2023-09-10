""" Print host name, driver, CUDA version, OS version, python version, python package versions """
from .nsight_utils import (
    print_system_info,
    print_python_env_info,
    print_conda_envs_info,
)

if __name__ == "__main__":
    from .detect_pwd import GRAPHILER_CONDA_ENV_NAME, HET_CONDA_ENV_NAME

    print_system_info()
    print_python_env_info()

    print("Conda graphiler envs:", GRAPHILER_CONDA_ENV_NAME)
    print("Conda het envs:", HET_CONDA_ENV_NAME)
    print(" ")

    print_conda_envs_info([GRAPHILER_CONDA_ENV_NAME, HET_CONDA_ENV_NAME])
