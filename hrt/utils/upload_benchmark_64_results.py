from .upload_benchmark_results import upload_folder

if __name__ == "__main__":
    from .detect_pwd import is_pwd_het_dev_root

    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"

    upload_folder("misc/artifacts", "benchmark_64_", False, False)
