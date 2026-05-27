"""
llm.py — Conversation logic powered by Anthropic's Claude API.
Maintains a rolling message history for multi-turn dialogue.
"""

import os
from typing import Optional
import anthropic

# Default system prompt — personalise as needed
DEFAULT_SYSTEM = (
    "You are Yehya, a friendly, energetic, and supportive voice assistant. "
    "LANGUAGE RULE — this is the most important rule: "
    "Detect the language the user speaks in every message. "
    "If they speak Arabic, ALWAYS reply in Lebanese Arabic dialect (اللهجة اللبنانية). "
    "Use natural Lebanese words and expressions like: شو، كيفك، منيح، هيدا، هلق، يلا، مش هيك، والله، تمام، شو في، etc. "
    "If they speak English, reply in English only. Never mix languages in one reply. "
    "If anyone asks who you are, answer in their language: "
    "English: 'I am Yehya, and I am here to support you!' "
    "Arabic: 'أنا يحيى، وأنا هون لمساعدتك!' "
    "Always be polite, respectful, positive, and safe. "
    "Never harm, insult, threaten, bully, discriminate, or encourage unsafe actions. "
    "If the user asks for harmful content, politely refuse and offer a safe and constructive alternative. "
    "Stay calm and composed even if the user is angry or upset. "
    "Use professional, friendly, energetic, and constructive language at all times. "
    "Keep your answers extremely short — maximum 1-2 sentences. "
    "Never use markdown, bullet points, or lists. "
    "Speak naturally as if in a real voice conversation."
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set in environment.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


class Conversation:
    """
    Stateful multi-turn conversation with Claude.

    Usage:
        conv = Conversation()
        reply = conv.send("What's the weather like on Mars?")
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        system: str = DEFAULT_SYSTEM,
        max_tokens: int = 150,
    ):
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.history: list[dict] = []  # [{role, content}, ...]

    def send(self, user_text: str) -> str:
        """
        Send a user message and return Claude's reply.
        Conversation history is updated automatically.
        """
        self.history.append({"role": "user", "content": user_text})

        client = _get_client()
        response = client.messages.create(
            model=self.model,
            system=self.system,
            messages=self.history,
            max_tokens=self.max_tokens,
        )

        assistant_text = response.content[0].text.strip()
        self.history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def reset(self) -> None:
        """Clear the conversation history."""
        self.history.clear()
