"""wmj.reporting — draws what the judge decided; decides nothing itself.

In plain words: this package turns verdict numbers into charts, captions
and (later) a static results page. It never computes a judgment metric
(reporting spec §4) — every number it draws is handed to it — and it is
the only package that writes files under `out/` (design-review-008 C8).
"""
