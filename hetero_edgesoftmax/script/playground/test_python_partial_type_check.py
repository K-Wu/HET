def func1(a, b: bool = True):
    print(a, b)


if __name__ == "__main__":
    func1(1, b=False)
    func1(1, b=2)
