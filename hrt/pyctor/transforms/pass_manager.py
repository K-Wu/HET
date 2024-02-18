from .passes import Pass
from ..ir.InterOpSSA.programs import Program
from ..ir.InterOpSSA.utils import RegistrySet
from typing import Callable
import sys


# This is a pointer to the module object instance itself.
# Reference: https://stackoverflow.com/a/35904211/5555077
this = sys.modules[__name__]
this.passes: RegistrySet[
    Callable
] = RegistrySet()  # Stores analysis and transform passes


class PassManager:
    def __init__(self, passes: list[Pass] = []):
        self.pass_pipeline: list[Pass] = passes
        self.done_and_valid_passes: set[str] = set()
        self.str_to_pass: dict[str, type[Pass]] = {}
        self.running_log: list[tuple[str, ...]] = []

    def add_pass(self, pass_: Pass) -> None:
        self.pass_pipeline.append(pass_)

    def add_passes(self, passes: list[Pass]) -> None:
        for pass_ in passes:
            self.add_pass(pass_)

    def register_str_to_pass(self, pass_: type[Pass]) -> None:
        self.str_to_pass[pass_.get_name()] = pass_

    def register_strs_to_passes(self, passes: list[type[Pass]]) -> None:
        for pass_ in passes:
            self.register_str_to_pass(pass_)

    def run_pass(self, pass_: Pass, program: Program):
        if pass_.get_name() in self.done_and_valid_passes:
            self.running_log.append(("skipped", pass_.get_name()))
            return

        prerequisites: list[str] = pass_.get_prerequisites(program)
        # Run the prerequisites that have no valid results
        for prereq_str in prerequisites:
            if prereq_str not in self.done_and_valid_passes:
                prereq = self.str_to_pass[prereq_str]()
                self.running_log.append(
                    (
                        "prerequisite",
                        pass_.get_name() + "<-" + prereq.get_name(),
                    )
                )
                self.run_pass(prereq, program)
        # Invalidate the results of the pass if it has been run before
        invalidated_passes: list[str] = pass_.run(program)
        self.running_log.append(("runned", pass_.get_name()))
        for pass_str_ in invalidated_passes:
            if pass_str_ in self.done_and_valid_passes:
                self.done_and_valid_passes.remove(pass_str_)
                self.running_log.append(
                    ("invalidated", pass_.get_name() + "->" + pass_str_)
                )
        self.done_and_valid_passes.add(pass_.get_name())

    def run(self, program: Program):
        # Inspired by run() in https://llvm.org/doxygen/PassManager_8h_source.html
        for pass_ in self.pass_pipeline:
            self.run_pass(pass_, program)
