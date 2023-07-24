import os
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
def nsys_exists() -> bool:
    """Check if nsys is installed."""
    return os.system("nsys --version >/dev/null 2>/dev/null") == 0


if __name__ == "__main__":
    assert nsys_exists() == True, "nsys is not installed"
    assert nsys_exists() == True, "nsys is not installed"
