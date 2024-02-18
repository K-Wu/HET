from ..ir.InterOpSSA.programs import Program
import abc


class Pass(metaclass=abc.ABCMeta):
    @classmethod
    def get_name(cls) -> str:
        # We didn't use class property because it is deprecated in Python 3.9
        raise NotImplementedError

    @abc.abstractmethod
    def get_prerequisites(self, program: Program) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, program: Program) -> list[str]:
        raise NotImplementedError
