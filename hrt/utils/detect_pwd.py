from .nsight_utils import get_git_root_path, get_env_name_from_setup
from .nsight_utils import create_new_results_dir as _create_new_results_dir
import os
import subprocess


def get_het_root_path() -> str:
    # go to the root path of the git repository, and go to the parent directory until we find the HET root path
    path = get_git_root_path()
    while not is_het_root_path(path):
        path = os.path.dirname(path)
    return path


def is_het_root_path(path: str) -> bool:
    try:
        # Check if we can find title as HET in README.md
        # File not found error during cat cannot be catched. So we use os.path.exists to check first
        if not os.path.exists(os.path.join(path, "README.md")):
            return False
        res = subprocess.check_output(["cat", os.path.join(path, "README.md")])
        res = res.decode("utf-8")
        if "# HET" in res:
            return True
        elif "## What's in a name?" in res:
            raise ValueError(
                "Fatal! Detected sub-header, What's in a name, in README.md"
                " but not found #HET. Is the top-level project renamed? Please"
                " update it in the detect_pwd.py, or avoid using the"
                " subheading in a non-top-level project."
            )
        return False
    except OSError:
        return False


def is_pwd_het_dev_root() -> bool:
    # return if pwd is get_het_root_path()/hrt
    return (
        is_het_root_path(os.path.dirname(os.getcwd()))
        and os.path.basename(os.getcwd()) == "HET"
    )


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
    return _create_new_results_dir(prefix, RESULTS_DIR)


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
    print("is het root path:", is_het_root_path(get_git_root_path()))
    print("het root path:", get_het_root_path())
    print("graphiler conda env name:", GRAPHILER_CONDA_ENV_NAME)
    print("het conda env name:", HET_CONDA_ENV_NAME)
