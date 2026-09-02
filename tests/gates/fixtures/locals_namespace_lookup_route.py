"""Evasion fixture: reaching a computed name via locals(), not getattr.

Independent-review finding, P1-C03 pass 4: locals() is the third
member of Python's namespace-introspection trio (locals/globals/vars)
-- globals and vars were already banned, but locals() alone reaches
the identical computed-string-lookup capability. Confirmed by
execution to reach math.sqrt using zero identifiers that were banned
before this fixture was added.
"""

import math


def evade():
    sqrt = math.sqrt
    ns = locals()
    fn = ns["s" + "qrt"]
    return fn(16)
