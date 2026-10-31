"""Rebuilding the exact prompt a provider's model saw.

A black-box audit replays a reply on a referee, so the referee needs the same input tokens: the chat template
applied to the request. `render_chat` applies the model's own Jinja template (shipped inside the GGUF, read with
the engine's `meta` command), the same template inference servers use. For gpt-oss (harmony format), the
reasoning the provider returned is spliced in before the answer so the referee scores the answer in the same
context the provider's model had.

Template drift is a real source of false alarms: if the provider used a different template or added a system
prompt, every token looks a little off. The audit checks it two ways: the provider-reported prompt token count
against ours, and the first answer tokens' agreement with the referee.
"""

from __future__ import annotations

import datetime as _dt
import functools
from typing import List, Optional

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _raise(msg):
    raise jinja2.TemplateError(msg)


@functools.lru_cache(maxsize=32)
def _compiled(template: str):
    """Compiling a chat template takes tens of milliseconds (gpt-oss's is large): do it once per template."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **_: __import__("json").dumps(
        x, indent=indent, ensure_ascii=ensure_ascii)
    env.globals["raise_exception"] = _raise
    return env.from_string(template)


def render_chat(template: str, messages: List[dict], add_generation_prompt: bool = True,
                date: Optional[str] = None, **kwargs) -> str:
    day = _dt.date.fromisoformat(date) if date else _dt.date.today()
    return _compiled(template).render(messages=messages, add_generation_prompt=add_generation_prompt,
                                      bos_token="", eos_token="",
                                      strftime_now=lambda fmt: _dt.datetime.combine(day, _dt.time()).strftime(fmt),
                                      **kwargs)


HARMONY_SYSTEM = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                  "Knowledge cutoff: 2024-06\nCurrent date: {date}\n\nReasoning: {effort}\n\n"
                  "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>")


def harmony_context(prompt: str, reasoning: str, effort: str, date: str) -> str:
    """gpt-oss prompt up to the start of the final answer, including the provider's own reasoning."""
    return (HARMONY_SYSTEM.format(date=date, effort=effort) + f"<|start|>user<|message|>{prompt}<|end|>"
            f"<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
            "<|start|>assistant<|channel|>final<|message|>")


def harmony_prompt_only(prompt: str, effort: str, date: str) -> str:
    """gpt-oss prompt up to where the model starts its reasoning (to score the reasoning itself)."""
    return (HARMONY_SYSTEM.format(date=date, effort=effort) + f"<|start|>user<|message|>{prompt}<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>")
