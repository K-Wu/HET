#!/usr/bin/env python3
def func1(a, b, c):
    print(a, b, c)
    return a + b + c


def func1(a):
    print(a)
    return a


if __name__ == "__main__":
    func1(1, 2, 3)
    func1(1)
