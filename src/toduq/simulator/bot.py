"""The chatbot module under test.

Wraps any `LLMClient` as a turn-taking task-oriented bot: given a user turn and
the dialogue history, it produces a response (and can produce N samples, the
basis for sampling-based UQ). The offline `EchoBot` lets the simulator run with
no model — useful for input-based UQ metrics that score the user turn directly.
"""
from __future__ import annotations

from typing import Optional

from toduq.runners.base import EchoClient, GenConfig, LLMClient

_SYSTEM = ("You are a task-oriented dialogue assistant helping a user complete a "
           "booking/search task. Respond helpfully and concisely to the user's turn.")


class Chatbot:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or EchoClient()

    def _prompt(self, turn: str, history: list[str]) -> str:
        ctx = "\n".join(history[-6:])
        return f"Dialogue so far:\n{ctx}\n\nUser: {turn}\nAssistant:"

    def respond(self, turn: str, history: list[str], *, cfg: Optional[GenConfig] = None) -> str:
        return self.llm.generate(self._prompt(turn, history), system=_SYSTEM, cfg=cfg)

    def sample(self, turn: str, history: list[str], n: int, *,
               cfg: Optional[GenConfig] = None) -> list[str]:
        return self.llm.sample(self._prompt(turn, history), n, system=_SYSTEM, cfg=cfg)
