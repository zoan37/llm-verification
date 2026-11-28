"""Seeded audit prompts, so a public repo does not mean a public, recognisable prompt set.

A fixed prompt list is an audit fingerprint: an endpoint could route exactly those prompts to the real model.
`generate(seed, n)` composes ordinary-looking requests from task templates and fillers. Run audits with a fresh,
private seed and publish the seed (and so the prompts) only after the snapshot is collected; anyone can then
regenerate and check the prompts that were used.

    python prompts/generate.py --seed "$(openssl rand -hex 16)" --n 48 > prompts/private-run.json
"""

import argparse
import hashlib
import json
import random

TASKS = [
    "Write a {lang} function that {code_task}. Keep it short and add a docstring.",
    "Explain {concept} to {audience} in about {n} sentences.",
    "Summarize the main causes of {event} in {n} bullet points.",
    "Write a {form} about {subject}.",
    "Translate into {language}: '{sentence}'",
    "What are the main differences between {pair}?",
    "Give me {n} practical tips for {goal}, with one sentence of reasoning each.",
    "Solve step by step: {math}",
    "Draft a short, polite email to {recipient} about {email_topic}.",
    "Describe how {process} works in plain language.",
    "List {n} interesting facts about {subject}.",
    "Write a product description for {product}.",
    "Return a compact JSON object with keys `summary`, `tradeoffs` (list of {n} strings), and `recommendation` comparing {pair}.",
    "Identify {n} subtle bugs or edge cases when implementing a {lang} function that {code_task}.",
    "You are explaining {concept} during a code review for {audience}. Write {n} concise sentences.",
]
FILL = {
    "lang": ["Python", "JavaScript", "Go", "Rust", "C", "TypeScript", "Java"],
    "code_task": ["reverses the words in a sentence", "checks whether a string is a palindrome",
                  "returns the n-th Fibonacci number iteratively", "removes duplicates from a list while keeping order",
                  "counts word frequencies in a text", "merges two sorted arrays", "validates an email address roughly",
                  "flattens a nested list", "computes the median of a list", "converts Celsius to Fahrenheit"],
    "concept": ["recursion", "inflation", "photosynthesis", "a hash table", "public-key encryption", "compound interest",
                "the greenhouse effect", "supply and demand", "gradient descent", "how vaccines work", "entropy",
                "a blockchain", "plate tectonics", "the placebo effect", "opportunity cost"],
    "audience": ["a ten-year-old", "a new programmer", "a busy executive", "a high-school student", "my grandmother"],
    "n": ["three", "four", "five", "six"],
    "event": ["the French Revolution", "the 2008 financial crisis", "World War I", "the fall of the Roman Empire",
              "the Industrial Revolution", "the dot-com crash"],
    "form": ["haiku", "limerick", "short poem", "two-paragraph story opening", "motivational paragraph"],
    "subject": ["the ocean at night", "a city in the rain", "an old lighthouse", "autumn leaves", "a lost cat",
                "the first day of school", "octopuses", "the Moon", "coffee", "mountain hiking", "honeybees"],
    "language": ["French", "Spanish", "German", "Italian", "Portuguese"],
    "sentence": ["The meeting was moved to Thursday afternoon.", "Could you recommend a good restaurant nearby?",
                 "I forgot my umbrella at the station.", "We should leave early to avoid traffic.",
                 "The library closes at eight on weekdays."],
    "pair": ["TCP and UDP", "SQL and NoSQL databases", "renting and buying a home", "electric cars and petrol cars",
             "Python lists and tuples", "weather and climate", "viruses and bacteria"],
    "goal": ["sleeping better", "learning a new language", "staying focused while working from home",
             "saving money on groceries", "running a first 5k", "writing clearer emails"],
    "math": ["a train travels 240 km in 3 hours; what is its average speed in m/s?",
             "what is 15% of 360, then increased by 20%?", "solve 3x + 7 = 31",
             "a rectangle has perimeter 50 and width 10; what is its area?",
             "how many ways can 4 books be arranged on a shelf?"],
    "recipient": ["a colleague", "my landlord", "a professor", "a client", "the event organizer"],
    "email_topic": ["rescheduling a meeting", "a late payment", "asking for an extension", "a broken heater",
                    "thanking them for their help"],
    "process": ["a refrigerator", "GPS", "the water cycle", "a credit card payment", "noise-cancelling headphones",
                "a search engine", "the immune system"],
    "product": ["a stainless steel water bottle", "a pair of wool socks", "a desk lamp", "a hiking backpack",
                "a cast-iron pan"],
}


def generate(seed: str, n: int):
    rng = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest(), "big"))
    out, seen = [], set()
    while len(out) < n:
        t = rng.choice(TASKS)
        p = t
        for k, vals in FILL.items():
            while "{" + k + "}" in p:
                p = p.replace("{" + k + "}", rng.choice(vals), 1)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def prompt_set_digest(prompts) -> str:
    payload = json.dumps(list(prompts), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def commit_seed(seed: str, n: int) -> dict:
    """Create a pre-collection commitment over (seed, prompts) without revealing the seed."""
    prompts = generate(seed, n)
    return {
        "n": int(n),
        "seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "prompts_sha256": prompt_set_digest(prompts),
    }


def verify_revealed_seed(seed: str, commitment: dict) -> bool:
    """Verify after collection that a revealed seed matches the pre-published commitment."""
    n = int(commitment["n"])
    if hashlib.sha256(seed.encode("utf-8")).hexdigest() != commitment["seed_sha256"]:
        return False
    return prompt_set_digest(generate(seed, n)) == commitment["prompts_sha256"]


if __name__ == "__main__":
    import secrets
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", help="seed string (auto-generates 128-bit secret hex if omitted)")
    ap.add_argument("--drand", action="store_true", help="mix latest drand quicknet beacon randomness into the seed")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--commit-out", help="optional path to write pre-collection {seed_sha256, prompts_sha256} commitment")
    ap.add_argument("--verify-commit", help="verify --seed against an existing commitment JSON file")
    a = ap.parse_args()
    seed = a.seed or secrets.token_hex(16)
    if a.verify_commit:
        comm = json.loads(open(a.verify_commit, "r", encoding="utf-8").read())
        ok = verify_revealed_seed(seed, comm)
        print(json.dumps({"verified": ok, "seed_sha256": comm.get("seed_sha256"), "prompts_sha256": comm.get("prompts_sha256")}))
        raise SystemExit(0 if ok else 1)
    if a.drand:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from llmverify.beacon import fetch_drand_beacon
        b = fetch_drand_beacon()
        seed = f"{seed}:drand:{b['round']}:{b['randomness']}"
    prompts = generate(seed, a.n)
    if a.commit_out:
        with open(a.commit_out, "w", encoding="utf-8") as f:
            json.dump(commit_seed(seed, a.n), f, indent=2)
    print(json.dumps(prompts, indent=1))
