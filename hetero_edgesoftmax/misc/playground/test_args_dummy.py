import argparse as A
import argparse as B


def ArgumentParser():
    return None


def try_intercept2():
    parser = A.ArgumentParser()
    if parser is None:
        print("parser is None")
    else:
        print("parser is not None")
