"""
whatsapp.py — Send WhatsApp messages via Twilio.

Environment variables required:
  TWILIO_ACCOUNT_SID   — Twilio Account SID
  TWILIO_AUTH_TOKEN    — Twilio Auth Token
  TWILIO_WHATSAPP_FROM — Sender number, e.g. whatsapp:+14155238886
  WHATSAPP_DEFAULT_TO  — Default recipient, e.g. whatsapp:+971501512255

Optional — named contacts (comma-separated name:number pairs):
  WHATSAPP_CONTACTS = "Mom:+971501234567,John:+9611234567,Ahmad:+9621234567"

  Say: "Send a WhatsApp to Mom saying I'll be late"
       "Send a WhatsApp to John saying hello"
       "Send a WhatsApp saying hello"  ← uses WHATSAPP_DEFAULT_TO
"""

import json
import os
import re
from pathlib import Path
from twilio.rest import Client

# contacts.json lives next to whatsapp.py
_CONTACTS_FILE = Path(__file__).parent / "contacts.json"


def _get_client() -> Client:
    sid   = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        raise EnvironmentError(
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set."
        )
    return Client(sid, token)


def load_contacts() -> dict[str, str]:
    """
    Load named contacts — checks contacts.json first, then WHATSAPP_CONTACTS env var.
    Returns dict of {lowercase_name: "whatsapp:+number"}
    """
    contacts = {}

    # 1. Load from contacts.json (managed via /contacts page)
    if _CONTACTS_FILE.exists():
        try:
            data = json.loads(_CONTACTS_FILE.read_text())
            for name, number in data.items():
                if not number.startswith("whatsapp:"):
                    number = f"whatsapp:{number}"
                contacts[name.lower()] = number
            return contacts
        except Exception:
            pass

    # 2. Fall back to env var
    raw = os.environ.get("WHATSAPP_CONTACTS", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            parts = entry.split(":", 1)
            name   = parts[0].strip().lower()
            number = parts[1].strip()
            if not number.startswith("whatsapp:"):
                number = f"whatsapp:{number}"
            contacts[name] = number
    return contacts


def resolve_recipient(name: str | None) -> str:
    """
    Resolve a contact name to a whatsapp:+number string.
    Falls back to WHATSAPP_DEFAULT_TO if name is None or not found.
    Raises EnvironmentError if no number can be resolved.
    """
    contacts = load_contacts()

    if name:
        key = name.strip().lower()
        if key in contacts:
            print(f"📋 Resolved contact '{name}' → {contacts[key]}")
            return contacts[key]
        else:
            print(f"⚠️  Contact '{name}' not found in contacts list. Using default.")

    default = os.environ.get("WHATSAPP_DEFAULT_TO")
    if not default:
        raise EnvironmentError(
            f"Contact '{name}' not found and WHATSAPP_DEFAULT_TO is not set."
        )
    if not default.startswith("whatsapp:"):
        default = f"whatsapp:{default}"
    return default


def send_whatsapp(message: str, to: str | None = None, recipient_name: str | None = None) -> str:
    """
    Send a WhatsApp message.

    Args:
        message:        Text body to send.
        to:             Explicit 'whatsapp:+XXXXXXXXXXX' — skips contact lookup.
        recipient_name: Contact name to look up (e.g. "Mom", "John").
                        Ignored if `to` is provided.

    Returns:
        Twilio message SID on success.
    """
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    to_number   = to or resolve_recipient(recipient_name)

    client = _get_client()
    msg = client.messages.create(
        from_=from_number,
        to=to_number,
        body=message,
    )
    print(f"📱 WhatsApp sent → {to_number} | SID: {msg.sid}")
    return msg.sid


def list_contacts() -> list[str]:
    """Return list of configured contact names."""
    return list(load_contacts().keys())
