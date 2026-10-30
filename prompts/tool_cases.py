"""Tool-calling cases: each has the tools offered, a user message, and what a correct reply must contain.

Grading (`grade`) is exact and needs no referee:
  * expect_call  a tool call to this function must be returned (not text)
  * args         required argument values (strings compared case-insensitively after trimming; numbers exactly)
  * no_call      no tool call should be made (the question needs none)
Every returned call must also name an offered function and carry arguments that parse as JSON and satisfy the
schema's required fields and basic types.
"""

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}}}
CONVERT = {"type": "function", "function": {
    "name": "convert_currency", "description": "Convert an amount between currencies.",
    "parameters": {"type": "object", "properties": {
        "amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO code"},
        "to_currency": {"type": "string", "description": "ISO code"}},
        "required": ["amount", "from_currency", "to_currency"]}}}
EVENT = {"type": "function", "function": {
    "name": "create_event", "description": "Create a calendar event.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
        "start_time": {"type": "string", "description": "HH:MM, 24-hour"},
        "duration_minutes": {"type": "integer"},
        "attendees": {"type": "array", "items": {"type": "string", "description": "email address"}}},
        "required": ["title", "date", "start_time", "duration_minutes"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search_products", "description": "Search a store catalog.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"},
        "filters": {"type": "object", "properties": {
            "max_price": {"type": "number"}, "in_stock": {"type": "boolean"},
            "category": {"type": "string", "enum": ["electronics", "books", "clothing", "home"]}}},
        "sort": {"type": "string", "enum": ["price_asc", "price_desc", "rating"]}},
        "required": ["query"]}}}
SQL = {"type": "function", "function": {
    "name": "run_sql", "description": "Run a read-only SQL query on the analytics database.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
EMAIL = {"type": "function", "function": {
    "name": "send_email", "description": "Send an email.",
    "parameters": {"type": "object", "properties": {
        "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
        "required": ["to", "subject", "body"]}}}
ALL = [WEATHER, CONVERT, EVENT, SEARCH, SQL, EMAIL]

CASES = [
    {"id": "weather_simple", "tools": [WEATHER], "user": "What's the weather like in Lisbon right now, in celsius?",
     "expect_call": "get_weather", "args": {"city": "Lisbon", "unit": "celsius"}},
    {"id": "weather_fahrenheit", "tools": [WEATHER], "user": "Is it cold in Chicago? Give it to me in Fahrenheit.",
     "expect_call": "get_weather", "args": {"city": "Chicago", "unit": "fahrenheit"}},
    {"id": "convert", "tools": [CONVERT], "user": "How much is 250 euros in Japanese yen?",
     "expect_call": "convert_currency", "args": {"amount": 250, "from_currency": "EUR", "to_currency": "JPY"}},
    {"id": "convert_decimal", "tools": [CONVERT], "user": "Convert 19.99 US dollars to British pounds.",
     "expect_call": "convert_currency", "args": {"amount": 19.99, "from_currency": "USD", "to_currency": "GBP"}},
    {"id": "event_full", "tools": [EVENT],
     "user": "Schedule 'Quarterly review' on 2026-11-12 at 14:30 for 45 minutes with ana@example.com and raj@example.com.",
     "expect_call": "create_event", "args": {"title": "Quarterly review", "date": "2026-11-12", "start_time": "14:30",
                                             "duration_minutes": 45, "attendees": ["ana@example.com", "raj@example.com"]}},
    {"id": "event_12h", "tools": [EVENT], "user": "Book 'Dentist' on 2026-12-03 at 9am, one hour.",
     "expect_call": "create_event", "args": {"title": "Dentist", "date": "2026-12-03", "start_time": "09:00",
                                             "duration_minutes": 60}},
    {"id": "search_nested", "tools": [SEARCH],
     "user": "Find noise-cancelling headphones under $200 that are in stock, cheapest first.",
     "expect_call": "search_products", "args": {"filters.max_price": 200, "filters.in_stock": True, "sort": "price_asc"}},
    {"id": "search_category", "tools": [SEARCH], "user": "Show me the highest-rated science fiction books.",
     "expect_call": "search_products", "args": {"filters.category": "books", "sort": "rating"}},
    {"id": "pick_weather", "tools": ALL, "user": "Do I need an umbrella in Seattle today? Use celsius.",
     "expect_call": "get_weather", "args": {"city": "Seattle", "unit": "celsius"}},
    {"id": "pick_convert", "tools": ALL, "user": "I have 1000 Swiss francs. What's that in euros?",
     "expect_call": "convert_currency", "args": {"amount": 1000, "from_currency": "CHF", "to_currency": "EUR"}},
    {"id": "pick_email", "tools": ALL,
     "user": "Email bob@example.com with subject 'Lunch' and say: Can we move lunch to 1pm?",
     "expect_call": "send_email", "args": {"to": "bob@example.com", "subject": "Lunch"}},
    {"id": "pick_sql", "tools": ALL, "user": "How many orders were placed yesterday? Query the orders table.",
     "expect_call": "run_sql", "args": {}},
    {"id": "no_call_math", "tools": ALL, "user": "What is 17 times 23?", "no_call": True},
    {"id": "no_call_chat", "tools": ALL, "user": "Write a two-line poem about autumn.", "no_call": True},
    {"id": "no_call_fact", "tools": [WEATHER], "user": "What is the capital of Australia?", "no_call": True},
    {"id": "unicode_city", "tools": [WEATHER], "user": "Weather in São Paulo, celsius please.",
     "expect_call": "get_weather", "args": {"city": "São Paulo", "unit": "celsius"}},
]


def _get(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _type_ok(v, schema):
    t = schema.get("type")
    ok = {"string": isinstance(v, str), "number": isinstance(v, (int, float)) and not isinstance(v, bool),
          "integer": isinstance(v, int) and not isinstance(v, bool), "boolean": isinstance(v, bool),
          "array": isinstance(v, list), "object": isinstance(v, dict)}.get(t, True)
    if ok and "enum" in schema:
        ok = v in schema["enum"]
    return ok


def grade(case: dict, message: dict) -> dict:
    """Returns {"ok": bool, "problems": [...]} for one assistant message (OpenAI format)."""
    import json
    calls = message.get("tool_calls") or []
    problems = []
    offered = {t["function"]["name"]: t["function"]["parameters"] for t in case["tools"]}
    parsed = []
    for c in calls:
        fn = (c.get("function") or {})
        name = fn.get("name")
        if name not in offered:
            problems.append(f"called an unknown function {name!r}")
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            problems.append(f"{name}: arguments are not valid JSON")
            continue
        schema = offered[name]
        for req in schema.get("required", []):
            if req not in args:
                problems.append(f"{name}: missing required argument {req!r}")
        for k, v in args.items():
            if k in schema.get("properties", {}) and not _type_ok(v, schema["properties"][k]):
                problems.append(f"{name}: argument {k!r} has the wrong type or value {v!r}")
        parsed.append((name, args))
    if case.get("no_call"):
        if calls:
            problems.append("made a tool call where none was needed")
    else:
        hits = [a for n, a in parsed if n == case["expect_call"]]
        if not calls:
            problems.append("answered in text instead of calling a tool"
                            + (" (but the text contains a call-like JSON)" if "{" in (message.get("content") or "") else ""))
        elif not hits:
            problems.append(f"called {[n for n, _ in parsed]} instead of {case['expect_call']}")
        else:
            args = hits[0]
            for k, want in case.get("args", {}).items():
                got = _get(args, k)
                if isinstance(want, str):
                    if not isinstance(got, str) or got.strip().lower() != want.lower():
                        problems.append(f"{k}: expected {want!r}, got {got!r}")
                elif isinstance(want, list):
                    if not isinstance(got, list) or sorted(map(str.lower, map(str, got))) != sorted(x.lower() for x in want):
                        problems.append(f"{k}: expected {want!r}, got {got!r}")
                elif got != want:
                    problems.append(f"{k}: expected {want!r}, got {got!r}")
    return {"ok": not problems, "problems": problems}
