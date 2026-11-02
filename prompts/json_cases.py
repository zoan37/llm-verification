"""Small structured-output cases with known answers, for experiments/reasoning_off.py.

Each case is a short news-style extraction or classification task (or one line of arithmetic) answered as one JSON
object. `expect` lists the keys that are graded exactly (strings case-insensitively, numbers to 1e-6); every key in
`keys` must be present.
"""

SYSTEM = ("You extract facts from short texts. Reply with one JSON object and nothing else, using exactly the keys "
          "requested. Use null when the text does not say.")

CASES = [
    {"q": 'Text: "Acme Corp reported third-quarter revenue of $412 million, up 8% from a year earlier." '
          'Keys: company (string), quarter (Q1-Q4), revenue_musd (number, millions of US dollars).',
     "keys": ["company", "quarter", "revenue_musd"], "expect": {"quarter": "Q3", "revenue_musd": 412}},
    {"q": 'Text: "The central bank raised its policy rate by 50 basis points to 4.25%." '
          'Keys: action ("hike", "cut" or "hold"), change_bp (number), new_rate_pct (number).',
     "keys": ["action", "change_bp", "new_rate_pct"], "expect": {"action": "hike", "change_bp": 50, "new_rate_pct": 4.25}},
    {"q": 'Text: "Shares of Norvel Bio fell 31% after regulators rejected its lead drug." '
          'Keys: direction ("up", "down" or "flat"), move_pct (number, unsigned), cause (string).',
     "keys": ["direction", "move_pct", "cause"], "expect": {"direction": "down", "move_pct": 31}},
    {"q": 'Text: "A fund holds 1,200 shares and sells a quarter of them, then buys 150 more." '
          'Keys: shares_now (number).',
     "keys": ["shares_now"], "expect": {"shares_now": 1050}},
    {"q": 'Text: "Brentwell Oil will cut output by 200,000 barrels per day starting in March, the company said." '
          'Keys: change ("increase", "decrease" or "none"), barrels_per_day (number), start_month (string).',
     "keys": ["change", "barrels_per_day", "start_month"],
     "expect": {"change": "decrease", "barrels_per_day": 200000, "start_month": "March"}},
    {"q": 'Text: "The exchange halted withdrawals for six hours after a security incident; no customer funds were lost." '
          'Keys: halted (true/false), hours (number), funds_lost (true/false).',
     "keys": ["halted", "hours", "funds_lost"], "expect": {"halted": True, "hours": 6, "funds_lost": False}},
    {"q": 'Text: "Ten-year yields rose to 4.62% from 4.48% on Tuesday." '
          'Keys: from_pct (number), to_pct (number), change_bp (number, signed).',
     "keys": ["from_pct", "to_pct", "change_bp"], "expect": {"from_pct": 4.48, "to_pct": 4.62, "change_bp": 14}},
    {"q": 'Text: "Quarterly earnings will be reported on 2026-11-19 after the close." '
          'Keys: date (YYYY-MM-DD), timing ("before_open", "after_close" or "unknown").',
     "keys": ["date", "timing"], "expect": {"date": "2026-11-19", "timing": "after_close"}},
    {"q": 'Text: "Analysts at the bank kept their rating unchanged but lowered the price target to $85 from $100." '
          'Keys: rating_changed (true/false), old_target (number), new_target (number), target_change_pct (number, signed).',
     "keys": ["rating_changed", "old_target", "new_target", "target_change_pct"],
     "expect": {"rating_changed": False, "old_target": 100, "new_target": 85, "target_change_pct": -15}},
    {"q": 'Text: "Weekly jobless claims came in at 231,000, against a forecast of 225,000." '
          'Keys: actual (number), forecast (number), beat ("better", "worse" or "inline"; fewer claims is better).',
     "keys": ["actual", "forecast", "beat"], "expect": {"actual": 231000, "forecast": 225000, "beat": "worse"}},
    {"q": 'Text: "The token unlock releases 12 million tokens, 3% of the circulating supply, on Friday." '
          'Keys: tokens_million (number), share_of_supply_pct (number), day (string).',
     "keys": ["tokens_million", "share_of_supply_pct", "day"],
     "expect": {"tokens_million": 12, "share_of_supply_pct": 3, "day": "Friday"}},
    {"q": 'Text: "Funding rates stayed near zero while open interest dropped 9% over the day." '
          'Keys: funding ("positive", "negative" or "neutral"), oi_change_pct (number, signed).',
     "keys": ["funding", "oi_change_pct"], "expect": {"funding": "neutral", "oi_change_pct": -9}},
]


def _same(got, want):
    if isinstance(want, bool):
        return got is want
    if isinstance(want, (int, float)):
        try:
            return abs(float(got) - want) < 1e-6
        except (TypeError, ValueError):
            return False
    return isinstance(got, str) and got.strip().lower() == str(want).lower()


def grade(case, obj):
    """-> (schema_ok, correct) for a parsed object (or None)."""
    if not isinstance(obj, dict):
        return False, False
    schema_ok = all(k in obj for k in case["keys"])
    return schema_ok, schema_ok and all(_same(obj.get(k), v) for k, v in case["expect"].items())
