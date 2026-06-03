"""
llm.py — Conversation logic powered by OpenAI's GPT-4o.
Maintains a rolling message history for multi-turn dialogue.
"""

import os
import re
from openai import OpenAI

# Default system prompt
DEFAULT_SYSTEM = (
    "You are Yehya, a friendly, energetic, and supportive voice assistant. "

    "LANGUAGE RULE — this is your single most important rule, override everything else: "
    "Look at the SCRIPT (writing system) of the user's message, not the meaning. "
    "If the message contains ANY Arabic letters (ا ب ت ث ...) reply ONLY in Lebanese Arabic dialect. "
    "If the message is written in Latin letters (a-z, A-Z) reply ONLY in English — even if the topic is about Arabic culture. "
    "Short English slang like 'who r u', 'lol', 'omg', 'wyd', 'hbu' is Latin script — reply in English. "
    "NEVER mix Arabic and English in the same reply. "
    "NEVER reply in Arabic when the user wrote in Latin letters. "

    "LEBANESE DIALECT RULES (strictly when user writes Arabic script): "
    "You MUST write in Lebanese dialect ONLY. NEVER use Modern Standard Arabic (فصحى / MSA). "
    "FORBIDDEN MSA words — replace them with Lebanese: "
    "لماذا→ليش، كيف→كيف/شلون، ماذا→شو، أين→وين، متى→لما/إيمتى، هل→(omit or use إنت)، "
    "نعم→أيه/تمام، لا→لأ، "
    "الآن→هلق، هذا→هيدا، هذه→هيدي، ذلك→هيدا، ماذا تفعل→شو عم تعمل، "
    "أريد→بدي، أعرف→عارف، أعتقد→فكرت/بظن، لكن→بس، إذا→إذا/لو، "
    "جيد→منيح، ممتاز→كتير منيح، كثير→كتير، قليل→شوي، ربما→يمكن، "
    "أيضاً→كمان، بالطبع→طبعاً، شكراً→يسلمو/مرسي، من فضلك→لو سمحت/إذا بتحب. "
    "Use authentic Lebanese filler words and expressions naturally: "
    "والله، يسلمو، يلا، مش هيك؟، هيك بيصير، لا2 جد؟، شو بدك، ما في شي، "
    "بالزبط، كتير منيح، يعني، تعبت منو، شو في ما في، هلق بحكيلك، "
    "بتعرف شو، روق، تمشي، عادي، بيكفي هيك، مو مشكلة. "
    "Greetings in Lebanese: مرحبا، كيفك، كيفكن، شو أخبارك، شو في؟، "
    "شو بيصير؟، كيف الحال؟، الحمدلله منيح. "

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

_ARABIC_RE = re.compile(r'[؀-ۿ]')

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment.")
        _client = OpenAI(api_key=api_key)
    return _client


class Conversation:
    """
    Stateful multi-turn conversation with GPT-4o.

    Usage:
        conv = Conversation()
        reply = conv.send("What's the weather like on Mars?")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        system: str = DEFAULT_SYSTEM,
        max_tokens: int = 200,
    ):
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.history: list[dict] = []  # [{role, content}, ...]

    def send(self, user_text: str) -> str:
        """
        Send a user message and return GPT-4o's reply.
        Detects user script and appends a hard language instruction so the model
        never replies in the wrong language.
        """
        is_arabic = bool(_ARABIC_RE.search(user_text))
        if is_arabic:
            lang_instruction = (
                "\n\n[SYSTEM: The user wrote in Arabic script. "
                "Reply ONLY in Lebanese Arabic dialect — the dialect spoken in Beirut and Mount Lebanon. "
                "Use words like: هلق، هيدا، هيدي، بدي، منيح، كتير، شو، وين، ليش، يلا، بس، كمان، يعني، والله، مرسي، يسلمو. "
                "NEVER use Modern Standard Arabic (فصحى). NEVER use Egyptian, Gulf, or Syrian dialect words. "
                "Do NOT use any English words in this reply.]"
            )
        else:
            lang_instruction = "\n\n[SYSTEM: The user wrote in Latin script (English). Reply ONLY in English. Do NOT use any Arabic.]"

        # Store original text in history
        self.history.append({"role": "user", "content": user_text})

        # Build messages with system prompt + language instruction injected into last user turn
        messages = (
            [{"role": "system", "content": self.system}]
            + self.history[:-1]
            + [{"role": "user", "content": user_text + lang_instruction}]
        )

        client = _get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        assistant_text = response.choices[0].message.content.strip()
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
        is_arabic = bool(_ARABIC_RE.search(assistant_reply))
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate short follow-up question suggestions for a voice assistant conversation."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=120,
        )
        lines = response.choices[0].message.content.strip().split("\n")
        suggestions = [l.strip().lstrip("-•123. ") for l in lines if l.strip()][:3]
        return suggestions
    except Exception as e:
        print(f"⚠️ Suggestions error: {e}")
        return []
