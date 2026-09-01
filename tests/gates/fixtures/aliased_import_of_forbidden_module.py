"""Evasion fixture: an aliased import of a module outside the allowlist."""

import os as o


def leaks_cwd():
    return o.getcwd()
