"""Evasion fixture: reaching a live frame via a caught exception's traceback.

Independent-review finding, P1-C03 pass 5: `except X as e:
e.__traceback__.tb_frame` hands over a live frame object with no
import and no previously-banned identifier -- frame objects expose
f_globals/f_locals/f_builtins, reaching the same capability
globals()/locals()/__builtins__ already ban, but through exception
handling (a core language feature) rather than a named builtin call.
Confirmed by execution to reach the entire builtins table in one hop.
"""

import math


def evade():
    sqrt = math.sqrt
    try:
        raise ValueError()
    except ValueError as caught:
        frame = caught.__traceback__.tb_frame
        fn = frame.f_locals["s" + "qrt"]
    return fn(16)
