"""Evasion fixture: reaching __builtins__ as a bare name reference.

(Not an ast.Attribute — __builtins__ here is an ast.Name, caught by
scan_banned_identifiers' Name branch, same as a call to eval() would be.)
"""


def leaks_eval():
    return __builtins__["eval"]
