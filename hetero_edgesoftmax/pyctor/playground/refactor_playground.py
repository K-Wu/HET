#!/usr/bin/env python3
#!/usr/bin/env python3
import ast
from refactor import Rule, Session, common
from refactor.actions import InsertAfter, Replace


class DummyRule(Rule):
    def match(self, node):
        # We are looking for
        # $result = run(...)
        assert isinstance(node, ast.For)
        # assert name
        # assert(node.target.id != "index_name")

        next_statement = common.next_statement_of(node, context=self.context)

        # assert $result != -1
        # create new_for node which changes the variable name to index_name
        new_target = ast.Name(id="index_name", ctx=ast.Store())
        new_for = ast.For(
            target=node.target,
            iter=node.iter,
            body=node.body,
            orelse=node.orelse,
            type_comment=node.type_comment,
        )
        # copy line_info from old node to new node
        print("abc")
        print(common.get_source_segment(self.context.source, node))
        print(common.position_for(node))
        new_for.lineno = node.lineno
        new_for.col_offset = node.col_offset
        new_for.end_lineno = node.end_lineno
        new_for.end_col_offset = node.end_col_offset

        # yield Replace(node, new_for)
        yield Replace(node.target, new_target)


module_0 = ast.parse(
    """
def main():
    for idx_i in range(10):
        for idx_j in range(15):
            result = do_something_else(idx_i, idx_j)
        do_something(result) # assumes result is good
"""
)

module_1 = ast.parse(
    """
def main():
    for idx_0 in range(10):
        for idx_i in range(10):
            for idx_j in range(15):
                result = do_something_else(idx_i, idx_j)
            do_something(result) # assumes result is good
"""
)

session = Session(rules=[DummyRule])
print(
    session.run(
        """
def main():
    for idx_i in range(10):
        for idx_j in range(15):
            result = do_something_else(idx_i, idx_j)
        do_something(result) # assumes result is good
"""
    )
)

pass
