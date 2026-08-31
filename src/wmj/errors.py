"""Shared base exception for the whole project.

In plain words: every error wmj raises is a WmjError underneath, and
every error message says three things — what failed, what was
expected, and which requirement's gate caught it — so a failure is
never a mystery (cross-cutting spec, Development Conventions).

This module holds only the shared base. Each package defines its own
concrete exceptions in its own errors.py, per the cross-cutting
convention: no central error-dumping ground, no re-export maze.
"""

from __future__ import annotations


class WmjError(Exception):
    """Base class for every exception this project raises.

    Not raised directly — subclassed per package (models, worlds,
    judge, harness, reporting), each in its own errors.py.
    """
