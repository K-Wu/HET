#!/usr/bin/env python3
def b(*args):
    print(args)


def a(*args):
    b(*args)


if __name__ == "__main__":
    a(1, 2, 3, 4, 5)
