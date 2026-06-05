"""
whatsapp.py — WhatsApp deep link integration.

Instead of sending via a third-party API, this module builds a wa.me deep link
that opens WhatsApp on the user's device with the contact and message pre-filled.
The user taps Send once — no third party, no API keys needed.

Contacts are loaded from contacts.json (managed via /contacts page) or
the WHATSAPP_CONTACTS env var (comma-separated Name:+number pairs).
"""

import json
import os
from urllib.parse import quote
from pathlib import Path

_CONTACTS_FILE = Path(__file__).parent / "contacts.json"


def load_contacts() -> dict[str, str]:
    """
    Load named contacts.
    Returns dict of {lowercase_name: "+number"}
    """
    contacts = {}

    if _CONTACTS_FILE.exists():
        try:
            data = json.loads(_CONTACTS_FILE.read_text())
            for name, number in data.items():
                # Strip whatsapp: prefix if present
                number = number.replace("whatsapp:", "").strip()
                contacts[name.lower()] = number
            return contacts
        except Exception:
            pass

    # Fall back to env var
    raw = os.environ.get("WHATSAPP_CONTACTS", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            parts  = entry.split(":", 1)
            name   = parts[0].strip().lower()
            number = parts[1].strip().replace("whatsapp:", "")
            contacts[name] = number

    return contacts


def resolve_number(recipient_name: str | None) -> str | None:
    """
    Resolve a contact name to a phone number.
    Returns None if not found and no default is set.
    """
    contacts = load_contacts()

    if recipient_name:
        key = recipient_name.strip().lower()
        if key in contacts:
            print(f"📋 Resolved '{recipient_name}' → {contacts[key]}")
            return contacts[key]
        else:
            print(f"⚠️  Contact '{recipient_name}' not found.")

    # Fall back to default
    default = os.environ.get("WHATSAPP_DEFAULT_TO", "").replace("whatsapp:", "").strip()
    return default if default else None


def build_deep_link(message: str, recipient_name: str | None = None, number: str | None = None) -> str:
    """
    Build a wa.me deep link that opens WhatsApp with a pre-filled message.

    If a number is provided it's used directly.
    Otherwise resolves from recipient_name or default contact.

    Returns the deep link URL string.
    """
    phone = number or resolve_number(recipient_name)
    encoded_msg = quote(message)

    if phone:
        # Remove any spaces or dashes from number
        phone = phone.replace(" ", "").replace("-", "")
        if not phone.startswith("+"):
            phone = "+" + phone
        return f"https://wa.me/{phone}?text={encoded_msg}"
    else:
        # No specific contact — opens WhatsApp with message, user picks contact
        return f"https://wa.me/?text={encoded_msg}"


def list_contacts() -> list[str]:
    """Return list of configured contact names."""
    return list(load_contacts().keys())
