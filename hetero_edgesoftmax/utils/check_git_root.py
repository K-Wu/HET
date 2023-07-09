#!/usr/bin/env python3
# some code are from in third_party/sputnik/codegen/utils.py
import subprocess
import os


def check_git_existence() -> None:
    """Check if git is installed and available in the path."""
    try:
        subprocess.check_output(["git", "--version"])
    except OSError:
        raise OSError("Git is not installed. Please install git and try again.")


def get_git_root_path() -> str:
    """Get the root path of the git repository."""
    check_git_existence()
    return os.path.normpath(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def is_het_root_path(path: str) -> bool:
    try:
        # Check if we can find title as HET in README.md
        res = subprocess.check_output(["cat", os.path.join(path, "README.md")])
        res = res.decode("utf-8")
        if "# HET" in res:
            return True
        return False
    except OSError:
        return False


if __name__ == "__main__":
    # hetero_edgesoftmax/third_party/sputnik$ python ../../hetero_edgesoftmax/utils/check_git_root.py
    # /home/kwu/hetero_edgesoftmax/third_party/sputnik
    # False

    # hetero_edgesoftmax/hetero_edgesoftmax$ python utils/check_git_root.py
    # /home/kwu/hetero_edgesoftmax
    # True
    print(get_git_root_path())
    print(is_het_root_path(get_git_root_path()))
