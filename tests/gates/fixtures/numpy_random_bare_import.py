"""Evasion fixture: bare import of numpy.random."""

import numpy.random


def draws():
    return numpy.random.normal(0, 1)
