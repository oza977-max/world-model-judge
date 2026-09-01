"""Evasion fixture: getattr(<allowlisted>, "__builtins__") reflection."""

import numpy as np


def leaks_eval():
    return getattr(np, "__builtins__")["eval"]
