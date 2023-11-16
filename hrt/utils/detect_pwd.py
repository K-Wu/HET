from .nsight_utils import (
    get_git_root_path,
    get_env_name_from_setup,
    is_generic_root_path,
    is_pwd_generic_dev_root,
    get_generic_root_path,
)
from .nsight_utils import create_new_results_dir as _create_new_results_dir
import os


def get_het_root_path() -> str:
    """Go to the root path of the git repository, and go to the parent directory until we find the HET root path"""
    return get_generic_root_path("HET")


def is_het_root_path(path: str) -> bool:
    return is_generic_root_path(path, "HET")


def is_pwd_het_dev_root() -> bool:
    """Return if pwd is get_het_root_path()/hrt"""
    return is_pwd_generic_dev_root("HET", "hrt")


GRAPHILER_CONDA_ENV_NAME = "graphiler"
# GRAPHILER_CONDA_ENV_NAME = "graphiler-new"
HET_CONDA_ENV_NAME = get_env_name_from_setup(get_het_root_path())


def is_conda_activated() -> bool:
    # Check if CONDA_SHLVL is set
    if "CONDA_SHLVL" in os.environ:
        return True
    return False


def get_conda_current_environment() -> str:
    # Check if CONDA_SHLVL is set
    assert (
        is_conda_activated()
    ), "Fatal! CONDA_SHLVL is not set. Is conda activated?"
    return os.environ["CONDA_DEFAULT_ENV"]


RESULTS_DIR = os.path.join("hrt", "misc", "artifacts")
RESULTS_ABSOLUTE_DIR = os.path.join(get_het_root_path(), RESULTS_DIR)


def create_new_results_dir(prefix: str) -> str:
    return _create_new_results_dir(prefix, RESULTS_ABSOLUTE_DIR)


if __name__ == "__main__":
    # HET/third_party/sputnik$ python ../../hrt/utils/check_git_root.py
    # git root path: /home/kwu/HET/third_party/sputnik
    # is het root path: False
    # het root path: /home/kwu/HET
    # graphiler conda env name: graphiler
    # het conda env name: graphiler

    # HET/hrt$ python utils/check_git_root.py
    # git root path: /home/kwu/HET
    # is het root path: True
    # het root path: /home/kwu/HET
    # graphiler conda env name: graphiler
    # het conda env name: graphiler
    print("git root path:", get_git_root_path())
    print(
        "is get_git_root_path returning het root path:",
        is_het_root_path(get_git_root_path()),
    )
    print("is_pwd_het_dev_root:", is_pwd_het_dev_root())
    print("het root path:", get_het_root_path())
    print("graphiler conda env name:", GRAPHILER_CONDA_ENV_NAME)
    print("het conda env name:", HET_CONDA_ENV_NAME)
