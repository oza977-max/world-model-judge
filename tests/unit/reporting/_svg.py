"""Reading a saved SVG chart's words, for the reporting tests.

In plain words: charts keep their text as text (`svg.fonttype: none`), but a sentence wrapped onto
several lines is split across several text elements. `svg_words` joins every text run in document
order and normalises the spacing, so a test can ask "does the image say this sentence?" whatever the
line breaks.
"""

from __future__ import annotations

import re


def svg_words(svg: str) -> str:
    return " ".join(" ".join(re.findall(r">([^<>]+)<", svg)).split())


def says(svg: str, sentence: str) -> bool:
    return " ".join(sentence.split()) in svg_words(svg)
