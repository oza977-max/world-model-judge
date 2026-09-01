"""Evasion fixture: a string-concatenated identifier.

**Honest limit, documented (independent-review finding, P1-C03 pass
1):** the concatenation itself (`"e" + "v" + "al"`) is invisible to an
identifier-based scan by construction — no Name/Attribute node ever
contains the literal string "eval". What this fixture actually proves
is narrower but still real: Python gives no way to turn a computed
string into a callable without going through one of the banned
invocation identifiers (`getattr`, `globals`, `vars`, `eval`,
`__import__`) — so a string-concatenation attack is still caught, but
via the identifier it's forced to invoke through
(`getattr`/`__builtins__` here), never via detecting the concatenation
route itself. This fixture is therefore confounded with the getattr/
reflection fixtures by the nature of the language, not by a gap in
this file — flagging that plainly rather than presenting it as
independent coverage of a distinct evasion class.
"""

_NAME = "e" + "v" + "al"


def runs_arbitrary_code():
    fn = getattr(__builtins__, _NAME)
    return fn("1 + 1")
