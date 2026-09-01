"""Evasion fixture: from-import of a numpy.random function."""

from numpy.random import rand


def draws():
    return rand()
