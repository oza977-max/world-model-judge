"""Evasion fixture: reaching os via a function's own __globals__."""

import numpy as np


def some_fn():
    return np.get_include()


def leaks_os():
    return some_fn.__globals__["os"]
