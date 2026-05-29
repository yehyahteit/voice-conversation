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

    "LANGUAGE RULE — this is your single most important rule, override everything else: "
    "Look at the SCRIPT (writing system) of the user's message, not the meaning. "
    "If the message contains ANY Arabic letters (ا ب ت ث ...) reply ONLY in Lebanese Arabic dialect. "
    "If the message is written in Latin letters (a-z, A-Z) reply ONLY in English — even if the topic is about Arabic culture. "
    "Short English slang like 'who r u', 'lol', 'omg', 'wyd', 'hbu' is Latin script — reply in English. "
    "NEVER mix Arabic and English in the same reply. "
    "NEVER reply in Arabic when the user wrote in Latin letters. "

    "Lebanese Arabic style (only when user writes Arabic script): "
    "Use natural Lebanese words: شو، كيفك، منيح، هيدا، هلق، يلا، مش هيك، والله، تمام، شو في. "

    "Identity: if asked who you are — "
    "in English say: 'I am Yehya, and I am here to support you!' "
    "in Arabic say: 'أنا يحيى، وأنا هون لمساعدتك!' "

    "Always be polite, respectful, positive, and safe. "
    "Never harm, insult, threaten, bully, discriminate, or encourage unsafe actions. "
    "Stay calm and composed even if the user is angry or upset. "
    "Keep answers extremely short — maximum 1-2 sentences. "
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
        max_tokens: int = 200,
    ):
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.history: list[dict] = []  # [{role, content}, ...]

    def send(self, user_text: str) -> str:
        """
        Send a user message and return Claude's reply.
        Detects user script and appends a hard language instruction so Claude
        never replies in the wrong language.
        """
        import re as _re
        is_arabic = bool(_re.search(r'[؀-ۿ؀-ۿ]', user_text))
        if is_arabic:
            lang_instruction = "\n\n[SYSTEM: The user wrote in Arabic script. Reply ONLY in Lebanese Arabic dialect. Do NOT use any English.]"
        else:
            lang_instruction = "\n\n[SYSTEM: The user wrote in Latin script (English). Reply ONLY in English. Do NOT use any Arabic.]"

        # Store original text in history, append instruction only for this API call
        self.history.append({"role": "user", "content": user_text})

        # Build messages with language instruction injected into last user turn
        messages_with_hint = self.history[:-1] + [
            {"role": "user", "content": user_text + lang_instruction}
        ]

        client = _get_client()
        response = client.messages.create(
            model=self.model,
            system=self.system,
            messages=messages_with_hint,
            max_tokens=self.max_tokens,
        )

        assistant_text = response.content[0].text.strip()
        self.history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def reset(self) -> None:
        """Clear the conversation history."""
        self.history.clear()


def generate_suggestions(assistant_reply: str, user_message: str) -> list[str]:
    """
    Generate 3 short follow-up question suggestions based on the last exchange.
    Returns a list of 3 strings, or empty list on failure.
    """
    try:
        client = _get_client()
        # Detect language from assistant reply
        is_arabic = bool(__import__('re').search(r'[؀-ۿ]', assistant_reply))
        lang_instruction = (
            "Reply in Lebanese Arabic dialect only." if is_arabic
            else "Reply in English only."
        )
        prompt = (
            f"The user said: \"{user_message}\"\n"
            f"The assistant replied: \"{assistant_reply}\"\n\n"
            f"Generate exactly 3 short follow-up questions the user might want to ask next. "
            f"Each must be under 8 words. {lang_instruction} "
            f"Return only the 3 questions, one per line, no numbering, no extra text."
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            system="You generate short follow-up question suggestions for a voice assistant conversation.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )
        lines = response.content[0].text.strip().split("\n")
        suggestions = [l.strip().lstrip("-•123. ") for l in lines if l.strip()][:3]
        return suggestions
    except Exception as e:
        print(f"⚠️ Suggestions error: {e}")
        return []
