"""Evasion fixture: getattr indirection reaching a banned attribute."""

import numpy as np


def leaks_builtins():
    return getattr(np, "__builtins__")
