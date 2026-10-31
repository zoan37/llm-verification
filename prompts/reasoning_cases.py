"""Reasoning-effort probes: problems with a single numeric answer, hard enough that more reasoning helps.

Used by experiments/reasoning_effort.py to check that an endpoint's reasoning effort setting (low/medium/high) is
honoured: reasoning length should grow with effort, accuracy should not fall, and billed reasoning tokens should match
the reasoning actually returned.
"""

CASES = [
    {"q": "A shop sells pens at 3 for $2 and notebooks at $5 each. Maya buys 18 pens and 4 notebooks and pays with a $50 bill. How much change does she get, in dollars?", "a": 18},
    {"q": "What is the sum of all positive divisors of 360?", "a": 1170},
    {"q": "How many integers between 1 and 1000 inclusive are divisible by 3 or 5 but not by 15?", "a": 401},
    {"q": "A train leaves at 09:47 and arrives at 14:23 the same day. How many minutes is the journey?", "a": 276},
    {"q": "In how many ways can 5 people sit around a round table if two particular people must sit next to each other?", "a": 12},
    {"q": "What is the remainder when 7^100 is divided by 13?", "a": 9},
    {"q": "A rectangle's length is 3 more than twice its width, and its perimeter is 66. What is its area?", "a": 230},
    {"q": "How many trailing zeros does 125! have?", "a": 31},
    {"q": "The average of five numbers is 18. Removing one number makes the average of the remaining four 16. What number was removed?", "a": 26},
    {"q": "How many 4-digit numbers have digits that sum to 9?", "a": 165},
    {"q": "A bag has 4 red and 6 blue balls. Two are drawn without replacement. The probability both are red is 1/n. What is n?", "a": 7.5},  # 2/15 = 1/7.5
    {"q": "What is the smallest positive integer that leaves remainder 2 when divided by 3, 4, 5 and 6, and is divisible by 7?", "a": 182},
]
SUFFIX = " End your answer with the final number on its own line, like: ANSWER: 42"


def grade(case: dict, content: str) -> bool:
    import re
    m = re.findall(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", content or "")
    if not m:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", content or "")
        m = nums[-1:] if nums else []
    if not m:
        return False
    try:
        return abs(float(m[-1].replace(",", "")) - case["a"]) < 1e-6
    except ValueError:
        return False
