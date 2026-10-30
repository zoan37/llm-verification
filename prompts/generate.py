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
    "What are the main differences between {a} and {b}?",
    "Give me {n} practical tips for {goal}, with one sentence of reasoning each.",
    "Solve step by step: {math}",
    "Draft a short, polite email to {recipient} about {email_topic}.",
    "Describe how {process} works in plain language.",
    "List {n} interesting facts about {subject}.",
    "Write a product description for {product}.",
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
    "a": ["TCP", "SQL databases", "renting", "electric cars", "Python lists", "weather", "viruses"],
    "b": ["UDP", "NoSQL databases", "buying a home", "petrol cars", "tuples", "climate", "bacteria"],
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    print(json.dumps(generate(a.seed, a.n), indent=1))
