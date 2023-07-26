#!/usr/bin/env python3
# some code are from in third_party/sputnik/codegen/utils.py
import subprocess
import os
import datetime
from functools import lru_cache


# From https://stackoverflow.com/a/4104188
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


@lru_cache(maxsize=None)
@run_once
def git_exists() -> None:
    """Check if git is installed and available in the path."""
    try:
        subprocess.check_output(["git", "--version"])
    except OSError:
        raise OSError("Git is not installed. Please install git and try again.")


def get_git_root_path() -> str:
    """Get the root path of the git repository."""
    git_exists()
    return os.path.normpath(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


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
                "Fatal! Detected sub-header, What's in a name, in README.md but not found #HET. Is the top-level project renamed? Please update it in the detect_pwd.py, or avoid using the subheading in a non-top-level project."
            )
        return False
    except OSError:
        return False


def is_pwd_het_dev_root() -> bool:
    # return if pwd is get_het_root_path()/hetero_edgesoftmax
    return (
        is_het_root_path(os.path.dirname(os.getcwd()))
        and os.path.basename(os.getcwd()) == "hetero_edgesoftmax"
    )


def get_env_name_from_setup() -> str:
    # read hetero_edgesoftmax/script/setup_dev_env.sh and get the conda env name
    setup_script_path = os.path.join(
        get_het_root_path(), "hetero_edgesoftmax", "script", "setup_dev_env.sh"
    )
    with open(setup_script_path, "r") as f:
        for line in f:
            if "conda activate" in line:
                return line.split(" ")[-1].strip()
    raise ValueError(
        "Fatal! Cannot find conda activate command in setup_dev_env.sh. Please check the file."
    )


GRAPHILER_CONDA_ENV_NAME = "graphiler"
# GRAPHILER_CONDA_ENV_NAME = "graphiler-new"
HET_CONDA_ENV_NAME = get_env_name_from_setup()


def is_conda_activated() -> bool:
    # Check if CONDA_SHLVL is set
    if "CONDA_SHLVL" in os.environ:
        return True
    return False


def get_conda_current_environment() -> str:
    # Check if CONDA_SHLVL is set
    assert is_conda_activated(), "Fatal! CONDA_SHLVL is not set. Is conda activated?"
    return os.environ["CONDA_DEFAULT_ENV"]


RESULTS_RELATIVE_DIR = os.path.join("hetero_edgesoftmax", "misc", "artifacts")
RESULTS_DIR = os.path.join(get_het_root_path(), RESULTS_RELATIVE_DIR)


def create_new_results_dir(prefix: str) -> str:
    curr_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    new_dir = os.path.join(RESULTS_DIR, prefix + curr_time)
    os.makedirs(new_dir)
    return new_dir


if __name__ == "__main__":
    # hetero_edgesoftmax/third_party/sputnik$ python ../../hetero_edgesoftmax/utils/check_git_root.py
    # git root path: /home/kwu/hetero_edgesoftmax/third_party/sputnik
    # is het root path: False
    # het root path: /home/kwu/hetero_edgesoftmax
    # graphiler conda env name: graphiler
    # het conda env name: graphiler

    # hetero_edgesoftmax/hetero_edgesoftmax$ python utils/check_git_root.py
    # git root path: /home/kwu/hetero_edgesoftmax
    # is het root path: True
    # het root path: /home/kwu/hetero_edgesoftmax
    # graphiler conda env name: graphiler
    # het conda env name: graphiler
    print("git root path:", get_git_root_path())
    print("is het root path:", is_het_root_path(get_git_root_path()))
    print("het root path:", get_het_root_path())
    print("graphiler conda env name:", GRAPHILER_CONDA_ENV_NAME)
    print("het conda env name:", HET_CONDA_ENV_NAME)