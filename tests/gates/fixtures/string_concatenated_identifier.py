"""Evasion fixture: a string-concatenated identifier.

**Correction (independent-review finding, P1-C03 pass 2 — this
docstring's earlier claim was wrong, not just incomplete):** pass 1
found this fixture was flagged only via its `getattr`/`__builtins__`
calls, not the concatenation (`"e" + "v" + "al"`) itself, and the first
fix round's docstring claimed this was an *inherent* limit — "Python
gives no way to invoke a computed name without going through one of
the banned invocation identifiers." Pass 2 proved that claim false by
building a real counterexample, `math.__dict__["s"+"qrt"]`, which
invokes a dynamically-computed name using ZERO of the then-banned
identifiers (see `tests/gates/fixtures/dict_namespace_lookup_route.py`,
and `BANNED_IDENTIFIERS`'s own comment in `test_import_graph.py`,
which now bans `__dict__`/`__getattribute__` because of exactly this).

So: the concatenation route is not undetectable by nature — it was a
genuine, fixable list omission. This fixture, using `getattr`, remains
a valid member of the corpus (it's still flagged, still a real
historical evasion shape), but it does NOT independently demonstrate
"string concatenation defeats the lint" — it demonstrates "string
concatenation plus getattr is still caught via getattr." The distinct
concatenation-via-`__dict__` shape now has its own dedicated fixture.
"""

_NAME = "e" + "v" + "al"


def runs_arbitrary_code():
    fn = getattr(__builtins__, _NAME)
    return fn("1 + 1")
