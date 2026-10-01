"""RAG Stage Group A / Stage 1: structured fact extraction.

The mandated Groww scheme pages are client-rendered Next.js apps. The rendered DOM
gives us prose but the numbers (expense ratio, exit load, lock-in, minimum SIP,
benchmark, riskometer) live in the public `__NEXT_DATA__` JSON payload embedded in
the same page. This module renders those public fields into clean, labelled fact
lines so they chunk and retrieve reliably.

The source of truth is still the public page - we are only choosing a different
serialisation of the same public content, not adding a new source.
"""

from __future__ import annotations

import json
import re
from typing import Any

_NEXT_DATA = re.compile(
    r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', re.S | re.I
)
# Some pages assign the payload as JS rather than emitting a JSON script tag.
_NEXT_DATA_ASSIGN = re.compile(
    r"(?:window|self)\.__NEXT_DATA__\s*=\s*(\{.*?\})\s*[;<]", re.S
)
_NEXT_DATA_ALT = re.compile(r"self\.__next_f\.push\(\[1,\s*\"(.*?)\"\]\)", re.S)
_ESCAPED_QUOTE = re.compile(r'\\"')


def extract_next_data(html: str) -> dict[str, Any] | None:
    """Best-effort parse of the embedded Next.js page payload."""
    match = _NEXT_DATA.search(html)
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            data = _dig_scheme_data(payload)
            if data:
                return data

    match = _NEXT_DATA_ASSIGN.search(html)
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            data = _dig_scheme_data(payload)
            if data:
                return data

    # App-router variant: the payload is pushed as escaped JSON string fragments.
    for fragment in _NEXT_DATA_ALT.findall(html):
        try:
            decoded = json.loads(f'"{_ESCAPED_QUOTE.sub(chr(34), fragment)}"')
        except json.JSONDecodeError:
            continue
        if '"mfServerSideData"' not in decoded:
            continue
        blob = decoded[decoded.index('{"pageProps"') :] if '{"pageProps"' in decoded else decoded
        for start in range(len(blob)):
            if blob.startswith('{"pageProps"', start):
                candidate, _ = _balanced_object(blob, start)
                if not candidate:
                    break
                try:
                    data = _dig_scheme_data(json.loads(candidate))
                except json.JSONDecodeError:
                    break
                if data:
                    return data
    return None


def _balanced_object(blob: str, start: int) -> tuple[str | None, int]:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(blob)):
        char = blob[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return blob[start : index + 1], index + 1
    return None, len(blob)


def _dig_scheme_data(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Locate mfServerSideData anywhere in the payload shape."""
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("mfServerSideData"), dict):
                return node["mfServerSideData"]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _lock_in_text(lock: Any) -> str | None:
    if not isinstance(lock, dict):
        return str(lock) if lock else None
    parts = []
    for unit, label in (("years", "year"), ("months", "month"), ("days", "day")):
        value = lock.get(unit)
        if isinstance(value, (int, float)) and value:
            parts.append(f"{int(value)} {label}{'s' if int(value) != 1 else ''}")
    return " ".join(parts) if parts else None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _aum_crore(data: dict[str, Any]) -> str | None:
    """Fund-level AUM in crore, or None.

    The page carries three different AUM numbers and only one of them is the fund's:

      - `aum` in the scheme object          -> the fund, e.g. 113606.47 Cr  <- wanted
      - `amc_info.aum`                      -> HDFC AMC's whole book, e.g. 986236.84 Cr
      - `peerComparison[].aum`              -> other funds' AUM, e.g. Bank of India's

    The page's own prose is a trap: it reads "The fund currently has an Asset Under
    Management(AUM) of Rs 9,86,237 Cr", which is the *AMC* figure, roughly 8.7x the fund's.
    Scraping that sentence would confidently report the wrong number, so this reads the
    structured field only, where the fund/AMC/peer distinction is already made by key.
    """
    raw = data.get("aum")
    if not isinstance(raw, (int, float)) or raw <= 0:
        return None
    # Indian numbering: lakhs, not millions. 113606.47 -> "1,13,606.47"
    return f"{raw:,.2f}"


_HOLDING_AS_OF_KEYS = ("portfolio_date", "as_on_date", "date")


def _holdings_as_of(data: dict[str, Any]) -> str:
    """The date the disclosed holdings are stated as of, if any."""
    for key in _HOLDING_AS_OF_KEYS:
        raw = data.get(key)
        if raw:
            return _short_date(raw)
    for item in data.get("holdings") or []:
        if isinstance(item, dict) and item.get("portfolio_date"):
            return _short_date(item["portfolio_date"])
    return ""


def _sum_holdings(data: dict[str, Any], field: str) -> list[tuple[str, float]]:
    """Sum `corpus_per` across holdings, grouped by `field`, largest first.

    The page does not publish a sector or asset-allocation table, but it does publish the
    holdings with a `sector_name` and a weight, which sum to exactly 100. So the split can
    be derived rather than invented - and it is labelled as derived, because a top-50
    disclosure is not the same thing as the fund's official allocation.
    """
    totals: dict[str, float] = {}
    for item in data.get("holdings") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get(field) or "").strip()
        try:
            weight = float(item.get("corpus_per") or 0)
        except (TypeError, ValueError):
            continue
        if not label or weight <= 0:
            continue
        totals[label] = totals.get(label, 0.0) + weight
    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


def _render_breakdown(pairs: list[tuple[str, float]], limit: int = 12) -> list[str]:
    if not pairs:
        return []
    rows = [f"- {label}: {weight:.2f}%" for label, weight in pairs[:limit]]
    shown = sum(weight for _, weight in pairs[:limit])
    total = sum(weight for _, weight in pairs)
    if len(pairs) > limit:
        rows.append(f"- ... {len(pairs) - limit} further line(s) omitted")
    elif shown < total - 0.01:
        rows.append(f"- other: {total - shown:.2f}%")
    return rows


def _render_holdings_facts(data: dict[str, Any], lines: list[str]) -> None:
    as_of = _holdings_as_of(data)
    suffix = f" (as of {as_of})" if as_of else ""

    sectors = _sum_holdings(data, "sector_name")
    if sectors:
        lines.append(f"## Sector allocation and exposure, from the fund's disclosed holdings{suffix}")
        lines.append("- Sector-wise weight in the portfolio:")
        lines.extend(_render_breakdown(sectors))

    assets = _sum_holdings(data, "instrument_name")
    if assets:
        # The wording matters for recall. People ask this as "equity debt cash allocation"
        # or as "the equity debt cash split", and the embedder is sensitive to which: with a
        # heading of "Asset composition by instrument" the block ranked 9th for "allocation"
        # and 1st for "split". Saying allocation, split, equity, debt and cash up front
        # brings it to the top for both.
        lines.append(f"## Asset allocation across equity, debt and cash (from disclosed holdings{suffix})")
        lines.append("- Portfolio split by asset type:")
        lines.extend(_render_breakdown(assets))

    top = [
        (str(item.get("company_name") or "").strip(), item.get("corpus_per"))
        for item in (data.get("holdings") or [])
        if isinstance(item, dict)
        and str(item.get("company_name") or "").strip()
        and str(item.get("instrument_name") or "").strip().lower() == "equity"
    ]
    if top:
        # Equity positions only. HDFC Small Cap's largest disclosed weight is "Repo" at
        # 10.25%, and answering "the top holding is Repo" is technically defensible and
        # practically useless. Repo and the debt lines are already in the asset block above.
        lines.append(f"## Top equity holdings{suffix}")
        for name, weight in top[:10]:
            try:
                lines.append(f"- {name}: {float(weight):.2f}%")
            except (TypeError, ValueError):
                continue


def _manager_lines(data: dict[str, Any]) -> list[str]:
    """Current fund manager(s), preferring the dated detail list over the scalar field.

    The page carries two answers and they disagree. For HDFC Large Cap, `fund_manager`
    says "Prashant Jain" - a manager who left years ago - while `fund_manager_details`
    lists Rahul Baijal (since 2022-07-28) and Dhruv Muchhal (since 2023-06-21), who are
    the managers now. The scalar is stale, so it is only used when there is no dated
    detail to fall back on. This is the same trap as the AUM field: the convenient value
    is the wrong one, and only the structured detail gets it right.
    """
    details = data.get("fund_manager_details")
    rows: list[tuple[str, str]] = []
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            person = _clean(item.get("person_name"))
            if not person:
                continue
            rows.append((person, _short_date(item.get("date_from")) if item.get("date_from") else ""))

    if rows:
        seen: set[str] = set()
        out: list[str] = []
        for person, since in rows:
            if person in seen:
                continue
            seen.add(person)
            out.append(f"- {person} (in role since {since})" if since else f"- {person}")
        return out

    manager = _clean(data.get("fund_manager"))
    return [f"- {manager} (as listed on the page; not dated)"] if manager else []


def render_scheme_facts(data: dict[str, Any], source_url: str) -> str:
    """Render the public scheme fields as labelled fact lines.

    Returns an empty string when none of the target fields are present, so a page
    without the payload simply contributes no facts.
    """
    plan = _clean(data.get("plan_type"))
    plan_suffix = f" ({plan} plan)" if plan else ""
    lines: list[str] = []

    def block(heading: str) -> list[str]:
        return [f"## {heading}"]

    def add(target: list[str], label: str, value: Any, *, plan_specific: bool = True) -> None:
        text = _clean(value)
        if text:
            suffix = plan_suffix if plan_specific else ""
            target.append(f"- {label}{suffix}: {text}")

    # The facts are split by topic rather than listed under one heading. A single block of
    # twenty unrelated fields embeds to a vector that is a poor match for any one of them:
    # "portfolio turnover ratio of HDFC Flexi Cap" ranked the combined block eighth, behind
    # three sibling chunks, so the answer never reached the model. Each topic below is a
    # block small enough to embed as itself.
    fees = block("Fees and exit load")
    add(fees, "Expense ratio (annual %)", data.get("expense_ratio"))
    add(fees, "Exit load", data.get("exit_load"))
    add(fees, "Lock-in period", _lock_in_text(data.get("lock_in")))
    add(fees, "Tax status", data.get("tax_status"))
    add(fees, "Stamp duty", data.get("stamp_duty"), plan_specific=False)
    if len(fees) > 1:
        lines.extend(fees)

    minimums = block("Minimum investment amounts")
    add(minimums, "Minimum SIP amount (INR)", data.get("min_sip_investment"))
    add(minimums, "Minimum lump-sum investment (INR)", data.get("min_investment_amount"))
    add(minimums, "Minimum withdrawal amount (INR)", data.get("min_withdrawal"))
    if len(minimums) > 1:
        lines.extend(minimums)

    risk = block("Benchmark and riskometer")
    add(risk, "Benchmark", data.get("benchmark_name") or data.get("benchmark"))
    add(risk, "Riskometer", data.get("nfo_risk"))
    add(risk, "Category", data.get("category"))
    add(risk, "Fund type", data.get("fund_type") or data.get("scheme_type"))
    if len(risk) > 1:
        lines.extend(risk)

    detail = block("Fund size and details")
    add(detail, "Fund size / AUM (INR crore)", _aum_crore(data))
    add(detail, "Fund house", data.get("fund_house"), plan_specific=False)
    add(detail, "Launch date", data.get("launch_date"), plan_specific=False)
    add(detail, "Portfolio turnover ratio (annual %)", data.get("portfolio_turnover"), plan_specific=False)
    for key, label in (("groww_rating", "Groww risk-reward rating (of 10)"),
                       ("crisil_rating", "CRISIL rating")):
        rating = _clean(data.get(key))
        if rating and rating.lower() not in ("null", "none"):
            detail.append(f"- {label}: {rating}")
    add(detail, "Registrar", data.get("registrar_agent"), plan_specific=False)
    add(detail, "Face value", data.get("face_value"), plan_specific=False)
    if data.get("sip_allowed") is False:
        detail.append("- SIP available: No")
    if len(detail) > 1:
        lines.extend(detail)

    managers = _manager_lines(data)
    if managers:
        # A heading of its own: HDFC Balanced Advantage lists six managers, and
        # "who manages this fund" then matches a heading rather than a buried bullet.
        lines.append("## Fund manager")
        lines.extend(managers)

    historic_rows: list[str] = []
    historic_exit = data.get("historic_exit_loads")
    if historic_exit:
        rendered = _render_historic(historic_exit, limit=3)
        if rendered:
            historic_rows.append("- Historic exit load changes:")
            historic_rows.extend(f"  {row}" for row in rendered)

    historic_expense = data.get("historic_fund_expense")
    if historic_expense:
        rendered = _render_historic(historic_expense, limit=3)
        if rendered:
            historic_rows.append("- Historic expense ratio changes:")
            historic_rows.extend(f"  {row}" for row in rendered)

    if historic_rows:
        # Its own heading: HDFC Large Cap alone carries 155 expense-ratio changes, which
        # would push the facts above them out of any single chunk and cost the "who manages
        # this fund" answer its heading.
        lines.append("## Historic changes to fees and exit load")
        lines.extend(historic_rows)

    # Derived blocks become `##` headings on purpose. They are long, and the chunker cuts
    # on headings: without one, sector exposure and top holdings would be glued into the
    # facts block, blow past max_chars and be fragmented, leaving sector percentages
    # separated from the heading that names them. A heading keeps each block retrievable.
    _render_holdings_facts(data, lines)

    # A payload with none of the target fields yields either no blocks or blocks that are
    # heading-only. Either way there is nothing to state, and returning a bare heading would
    # put a contentless chunk in the corpus.
    if not any(line.startswith("-") for line in lines):
        return ""
    lines.append(f"Source page: {source_url}")
    return "\n".join(lines)


_DATE_KEYS = ("as_on_date", "date", "from_date", "to_date", "effective_date")


def _render_historic(value: Any, limit: int = 8) -> list[str]:
    """Render a historic series as *changes only*.

    The published series is daily, so consecutive records almost always repeat.
    Emitting them raw bloated the corpus 11x with identical facts. A row is kept
    only when a non-date field actually changes; the date is then reported as the
    effective date of that change, which is the information the date carries.
    """
    items = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
    rows: list[str] = []
    previous: str | None = None
    for key, item in items:
        if isinstance(item, dict):
            fields_ = {
                k: _render_value(v)
                for k, v in item.items()
                if _clean(v) is not None and k not in _DATE_KEYS
            }
            if not fields_:
                continue
            signature = ", ".join(f"{k}={v}" for k, v in fields_.items())
            effective = next(
                (
                    _short_date(item[k])
                    for k in _DATE_KEYS
                    if k in item and _clean(item[k]) is not None
                ),
                _short_date(key),
            )
        elif _clean(item):
            signature, effective = str(_clean(item)), _short_date(key)
        else:
            continue
        if signature == previous:
            continue
        previous = signature
        rows.append(f"effective {effective}: {signature}")
    if len(rows) > limit:
        omitted = len(rows) - limit
        rows = rows[:limit] + [f"... {omitted} further change(s) omitted"]
    return rows


def _render_value(value: Any) -> str:
    text = str(_clean(value) or "")
    if text.endswith("T00:00:00"):
        text = text.split("T", 1)[0]
    return text


def _short_date(value: Any) -> str:
    text = str(value)
    return text.split("T", 1)[0] if "T" in text else text


_AUM_VALUE = re.compile(r'"aum\\?":\s*\\?"\s*(\d[\d,]*(?:\.\d+)?)')
_AUM_AS_OF = re.compile(
    r'"aumAsMonth\\?":\s*\\?"\s*\(?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
)


def extract_official_aum(html: str, source_url: str) -> str:
    """Pull the fund's AUM out of an AMC page's own payload.

    hdfcfund.com publishes it as an adjacent key pair in its Next.js payload:
    `"aum":"113,606.47","aumAsMonth":"(31/08/2026)"`. The rendered markup is a bare
    `<div>...AUM...<span>(31/08/2026)</span></div><div>â‚¹113,606.47 Cr.</div>`, which
    main-text extraction flattens or drops, so the number never reached the corpus even
    though the official page displays it. Reading the key pair keeps the official source
    authoritative for fund size instead of borrowing an aggregator's figure.

    The `aum`/`aumAsMonth` adjacency is what makes this safe: the AMC's own book is
    published under different keys, so a fund-level reading is the only one that matches.
    """
    value = _AUM_VALUE.search(html)
    if not value:
        return ""
    amount = value.group(1)
    try:
        if float(amount.replace(",", "")) <= 0:
            return ""
    except ValueError:
        return ""

    line = f"- Fund size / AUM (INR crore): {amount}"
    as_of = _AUM_AS_OF.search(html, value.end())
    if as_of:
        line += f" (as of {as_of.group(1)})"
    return f"Official page AUM (from the fund house's published data):\n{line}\nSource page: {source_url}"


def extract_facts_section(html: str, source_url: str) -> str:
    """Public entry point: HTML in, rendered fact lines (or '') out."""
    data = extract_next_data(html)
    facts = render_scheme_facts(data, source_url) if data else ""
    if "Fund size / AUM" in facts:
        return facts
    official = extract_official_aum(html, source_url)
    if not official:
        return facts
    return f"{facts}\n\n{official}" if facts else official

