"""Evasion fixture: a custom metaclass hands over the class-body
namespace dict as an ordinary parameter.

Independent-review finding, P1-C03 pass 3: this reaches the exact same
capability as __dict__/getattr (look something up by a computed name)
but touches zero banned identifiers — the namespace dict just arrives
as a normal argument to __new__. Caught by scan_metaclass_usage's
structural ban on `metaclass=`, not by the identifier scan.
"""

import math

_CAPTURED: list = []


class _NamespaceCapturingMeta(type):
    def __new__(mcs, name, bases, namespace):
        _CAPTURED.append(namespace)
        return super().__new__(mcs, name, bases, namespace)


class _Probe(metaclass=_NamespaceCapturingMeta):
    sqrt = math.sqrt


def evade():
    namespace = _CAPTURED[0]
    fn = namespace["sqrt"]
    return fn(16)
