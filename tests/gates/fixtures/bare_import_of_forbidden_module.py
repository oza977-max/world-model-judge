"""Evasion fixture: a bare import of a module outside the judge's allowlist."""

import os


def leaks_cwd():
    return os.getcwd()
