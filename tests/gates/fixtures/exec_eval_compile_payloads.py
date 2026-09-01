"""Evasion fixture: exec/eval/compile payloads."""


def runs_arbitrary_code():
    exec("import os")
    eval("1 + 1")
    return compile("1 + 1", "<string>", "eval")
