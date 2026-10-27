"""wmj.reporting — draws what the judge decided; decides nothing itself.

In plain words: this package turns verdict numbers into charts, captions
and a static results page. It never computes a judgment metric
(reporting spec §4) — every number it draws is handed to it — and its
writer (`writer.write_run`) is to be the only code that writes files under
`out/` (design-review-008 C8; `harness/skeleton.py`'s old write there is
owed a move at P6-C01).
"""
