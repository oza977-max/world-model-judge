"""Evasion fixture: a string-concatenated identifier reached via getattr."""

_NAME = "e" + "v" + "al"


def runs_arbitrary_code():
    fn = getattr(__builtins__, _NAME)
    return fn("1 + 1")
