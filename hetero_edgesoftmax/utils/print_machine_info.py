""" Print host name, driver, CUDA version, OS version, python version, python package versions """

import os
import socket


if __name__ == "__main__":
    print("Host name:", socket.gethostname())

    gpu_info = [
        item
        for item in os.popen("nvidia-smi | grep Driver").read().split("  ")[:-1]
        if len(item) > 0
    ]

    print("GPU driver:", gpu_info[-2])
    print("CUDA:", gpu_info[-1])

    ncu_version_info = [
        ele for ele in os.popen("ncu --version").read().split("\n") if len(ele) > 0
    ][-1].strip()
    nsys_version_info = [
        ele for ele in os.popen("nsys --version").read().split("\n") if len(ele) > 0
    ][-1].strip()
    print("NVIDIA NCU:", ncu_version_info)
    print("NVIDIA NSYS:", nsys_version_info)

    with open("/etc/os-release") as fd:
        for line in fd.readlines():
            if line.startswith("PRETTY_NAME="):
                print("OS:", line.split("=")[-1].strip())

    print("OS kernel version:", os.popen("uname -r").read().strip())
    print(" ")

    print("Python:", os.popen("python --version").read().strip())
    print("--------------Python packages--------------")
    print(os.popen("pip list").read())
    print("-------------------------------------------")
    print(" ")

    from .detect_pwd import GRAPHILER_CONDA_ENV_NAME, HET_CONDA_ENV_NAME

    print("Conda:", os.popen("conda --version").read().strip())
    print("-------Conda packages (graphiler)----------")
    print(os.popen("conda list -n " + GRAPHILER_CONDA_ENV_NAME).read())
    print("-------------------------------------------")
    print("-----------Conda packages (het)------------")
    print(os.popen("conda list -n " + HET_CONDA_ENV_NAME).read())
    print("-------------------------------------------")
