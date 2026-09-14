"""Work around a fast_flights 3.1.0 parser crash.

parser.parse_js() does `if payload[3][0] is None`, but payload[3] itself can be
None. Verified empirically: payload[3] is None exactly when Google returns no
nonstop option for that date (cross-checked against a control date beyond the
published schedule end, and against NRT-AOJ which shares the same booking
horizon limit).

Without this patch ~20% of queries die with an unexplained TypeError and you
cannot tell "no flight that day" from "fetch failed" — which silently corrupts
a longitudinal dataset. With it, no-flight days are recorded as an empty leg
list, which is the truth.
"""
import json
from selectolax.lexbor import LexborHTMLParser
from fast_flights import parser as _p


def _safe_parse(html: str) -> "_p.ResultList":
    script = LexborHTMLParser(html).css_first(r"script.ds\:1")
    if script is None:
        raise ValueError("no ds:1 script in html (blocked or layout change)")
    text = script.text()
    i = text.find("data:")
    if i < 0:
        raise ValueError("no data: payload in script")
    raw = text[i + 5:]
    raw = raw[:raw.rfind(", sideChannel")]
    payload = json.loads(raw)
    if payload[3] is None:          # genuine "no nonstop flight this date"
        rl = _p.ResultList()
        rl.metadata = _p.JsMetadata(alliances=[], airlines=[])
        return rl
    return _p.parse_js(text)


def apply() -> None:
    _p.parse = _safe_parse
    import fast_flights.fetcher as _f
    _f.parse = _safe_parse
