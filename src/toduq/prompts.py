"""Prompt loading + rendering.

Templates live as .txt files in the top-level `prompts/` directory so they can be
edited without touching code. Each pass and the judge have their own template.
"""
from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def render_paraphrase_prompt(utterance: str, n: int = 3) -> str:
    tmpl = load("apply_paraphrase.txt")
    return tmpl.replace("{{n}}", str(n)).replace("{{utterance}}", utterance)


def render_judge_prompt(change_from: str, change_to: str, intended: str, gold_action: str) -> str:
    tmpl = load("judge_validate.txt")
    return (
        tmpl.replace("{{change_from}}", change_from)
        .replace("{{change_to}}", change_to)
        .replace("{{intended_uncertainty}}", intended)
        .replace("{{gold_action}}", gold_action)
    )
