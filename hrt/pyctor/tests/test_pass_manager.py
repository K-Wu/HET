from ..transforms.pass_manager import PassManager
from ..transforms.passes import Pass
from ..ir.InterOpSSA.programs import Program
from ..ir.InterOpSSA.variable_tables import VariableTable


def create_empty_program():
    return Program(VariableTable(None), [])


class DepOnlyValueNumbererPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "DepOnlyValueNumbererPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        return []

    def run(self, program: Program) -> list[str]:
        return []


class DepOnlyDefUseAnalyzerPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "DepOnlyDefUseAnalyzerPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        return [DepOnlyValueNumbererPass.get_name()]

    def run(self, program: Program) -> list[str]:
        return []


class DepOnlyShapeInfererPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "DepOnlyShapeInfererPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        return [DepOnlyDefUseAnalyzerPass.get_name()]

    def run(self, program: Program) -> list[str]:
        return []


class DepOnlyBinopRealizerPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "DepOnlyBinopRealizerPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        return [DepOnlyShapeInfererPass.get_name()]

    def run(self, program: Program) -> list[str]:
        return [
            DepOnlyValueNumbererPass.get_name(),
            DepOnlyDefUseAnalyzerPass.get_name(),
            DepOnlyShapeInfererPass.get_name(),
        ]


def test0():
    pass_manager = PassManager()
    passes = [
        DepOnlyValueNumbererPass,
        DepOnlyDefUseAnalyzerPass,
        DepOnlyShapeInfererPass,
        DepOnlyBinopRealizerPass,
        DepOnlyValueNumbererPass,
        DepOnlyDefUseAnalyzerPass,
        DepOnlyShapeInfererPass,
    ]
    pass_manager.register_strs_to_passes(
        [
            DepOnlyValueNumbererPass,
            DepOnlyDefUseAnalyzerPass,
            DepOnlyShapeInfererPass,
            DepOnlyBinopRealizerPass,
        ]
    )
    pass_manager.add_passes([pass_() for pass_ in passes])
    pass_manager.run(create_empty_program())
    print("Now printing test0() logs")
    for log in pass_manager.running_log:
        print(log)


def test_print_empty_log():
    pass_manager = PassManager()
    print("Now printing test_print_empty_log() logs")
    for log in pass_manager.running_log:
        print(log)


def test1():
    pass_manager = PassManager()
    passes = [
        DepOnlyShapeInfererPass,
        DepOnlyBinopRealizerPass,
        DepOnlyValueNumbererPass,
        DepOnlyDefUseAnalyzerPass,
        DepOnlyShapeInfererPass,
    ]
    pass_manager.register_strs_to_passes(
        [
            DepOnlyValueNumbererPass,
            DepOnlyDefUseAnalyzerPass,
            DepOnlyShapeInfererPass,
            DepOnlyBinopRealizerPass,
        ]
    )
    pass_manager.add_passes([pass_() for pass_ in passes])
    pass_manager.run(create_empty_program())
    print("Now printing test1() logs")
    for log in pass_manager.running_log:
        print(log)


def test2():
    pass_manager = PassManager()
    passes = [
        DepOnlyShapeInfererPass,
        DepOnlyBinopRealizerPass,
        DepOnlyShapeInfererPass,
    ]
    pass_manager.register_strs_to_passes(
        [
            DepOnlyValueNumbererPass,
            DepOnlyDefUseAnalyzerPass,
            DepOnlyShapeInfererPass,
            DepOnlyBinopRealizerPass,
        ]
    )
    pass_manager.add_passes([pass_() for pass_ in passes])
    pass_manager.run(create_empty_program())
    print("Now printing test2() logs")
    for log in pass_manager.running_log:
        print(log)


def test3():
    pass_manager = PassManager()
    passes = [DepOnlyShapeInfererPass, DepOnlyBinopRealizerPass]
    pass_manager.register_strs_to_passes(
        [
            DepOnlyValueNumbererPass,
            DepOnlyDefUseAnalyzerPass,
            DepOnlyShapeInfererPass,
            DepOnlyBinopRealizerPass,
        ]
    )
    pass_manager.add_passes([pass_() for pass_ in passes])
    pass_manager.run(create_empty_program())
    print("Now printing test3() logs")
    for log in pass_manager.running_log:
        print(log)


if __name__ == "__main__":
    test0()
    test_print_empty_log()
    test1()
    test2()
    test3()
