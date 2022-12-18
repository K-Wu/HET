def my_func(a, b, c, d=1, e=2, f=3):
    print(a, b, c, d, e, f)


if __name__ == "__main__":
    my_func(4, 6, 7, f=5)
    my_func(4, 6, 7, 5)
