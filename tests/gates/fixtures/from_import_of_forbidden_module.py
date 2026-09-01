"""Evasion fixture: a from-import naming a specific forbidden function."""

from os import getcwd


def leaks_cwd():
    return getcwd()
