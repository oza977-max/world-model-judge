"""Evasion fixture: reaching a computed attribute via __dict__, not getattr.

Independent-review finding, P1-C03 pass 2: `module.__dict__[name]` is
a working bypass of the original 13-identifier list — it retrieves and
can call a dynamically-named attribute using only an allowlisted
import (`math`) and zero of the originally-banned identifiers. This
fixture is the reason `__dict__` and `__getattribute__` were added to
BANNED_IDENTIFIERS in tests/gates/test_import_graph.py.
"""

import math

_NAME = "s" + "qrt"


def evade():
    fn = math.__dict__[_NAME]
    return fn(16)
