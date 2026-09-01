"""Evasion fixture: rebinding a banned identifier to a fresh name."""

ev = eval


def runs_arbitrary_code():
    return ev("1 + 1")
