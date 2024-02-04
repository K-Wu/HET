from graphlib import TopologicalSorter
from .passes import Pass


class PassManager:
    def __init__(self):
        self.passes: list[Pass] = []

    def add_pass(self, pass_):
        self.passes.append(pass_)

    def run(self, module):
        for pass_ in self.passes:
            pass_.run(module)


if __name__ == "__main__":
    # From https://docs.python.org/3/library/graphlib.html#graphlib.TopologicalSorter
    graph = {"D": {"B", "C"}, "C": {"A"}, "B": {"A"}}
    ts = TopologicalSorter(graph)
    print(tuple(ts.static_order()))  # ('A', 'C', 'B', 'D')
