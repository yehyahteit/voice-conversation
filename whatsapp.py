"""
whatsapp.py — Send WhatsApp messages via Twilio.

Environment variables required:
  TWILIO_ACCOUNT_SID   — Twilio Account SID
  TWILIO_AUTH_TOKEN    — Twilio Auth Token
  TWILIO_WHATSAPP_FROM — Sender number, e.g. whatsapp:+14155238886
  WHATSAPP_DEFAULT_TO  — Default recipient, e.g. whatsapp:+971501512255
"""

import os
from twilio.rest import Client


def _get_client() -> Client:
    sid   = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        raise EnvironmentError(
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set."
        )
    return Client(sid, token)


def send_whatsapp(message: str, to: str | None = None) -> str:
    """
    Send a WhatsApp message.

    Args:
        message: Text body to send.
        to:      Recipient in 'whatsapp:+XXXXXXXXXXX' format.
                 Falls back to WHATSAPP_DEFAULT_TO env var.

    Returns:
        Twilio message SID on success.

    Raises:
        EnvironmentError: if credentials or recipient are missing.
        twilio.base.exceptions.TwilioRestException: on API failure.
    """
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    to_number   = to or os.environ.get("WHATSAPP_DEFAULT_TO")

    if not to_number:
        raise EnvironmentError(
            "Recipient not provided and WHATSAPP_DEFAULT_TO is not set."
        )

    client = _get_client()
    msg = client.messages.create(
        from_=from_number,
        to=to_number,
        body=message,
    )
    print(f"📱 WhatsApp sent → {to_number} | SID: {msg.sid}")
    return msg.sid
