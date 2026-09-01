"""Evasion fixture: aliasing importlib to dynamically reach a module."""

import importlib as il


def leaks_os():
    return il.import_module("os").getcwd()
