def is_pwd_het_dev_root() -> bool:
    """Use hrt/utils/detect_pwd.py to detect if current working directory is het_dev root directory."""
    try:
        from utils.detect_pwd import is_pwd_het_dev_root

        return is_pwd_het_dev_root()
    except ImportError:
        return False
