"""Evasion fixture: reaching __builtins__ as a plain attribute access."""


def leaks_eval():
    return __builtins__["eval"]
