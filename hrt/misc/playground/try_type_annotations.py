"""It is totally fine to have comments or docstring before the first-line from __future__ import annotations"""

from __future__ import annotations
import io
from typing import Union


def print_lines(lines: Union[list[str], io.TextIOWrapper]) -> None:
    for line in lines:
        print(line)


if __name__ == "__main__":
    print_lines(open("test.log"))
