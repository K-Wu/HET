#!/usr/bin/env python3
import ast
from refactor import Rule, Session, common
from refactor.actions import InsertAfter


class DummyRule(Rule):
    def match(self, node):
        # We are looking for
        # $result = run(...)
        assert isinstance(node, ast.Assign)

        # Check for = run(...)
        assert isinstance(call := node.value, ast.Call)
        assert isinstance(func := call.func, ast.Name)
        assert func.id == "run"

        # Ensure that we are only dealing with a simple target
        assert len(targets := node.targets) == 1
        assert isinstance(target := targets[0], ast.Name)

        # IMPORTANT: Ensure that the next statement is not
        # an assert already.
        next_statement = common.next_statement_of(node, context=self.context)
        assert next_statement is None or not isinstance(
            next_statement, ast.Assert
        )

        # assert $result != -1
        sanity_check = ast.Assert(
            test=ast.Compare(
                left=target, ops=[ast.NotEq()], comparators=[ast.Constant(-1)]
            )
        )

        return InsertAfter(node, sanity_check)


session = Session(rules=[DummyRule])
print(
    session.run(
        """
from very_dangerous import run

def main():
    result = run("very very problematic")
    do_something(result) # assumes result is good

result = run("something else")
do_something(result)
"""
    )
)
