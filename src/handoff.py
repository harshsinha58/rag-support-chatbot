"""
Human handoff: creates a 'ticket' (a JSON log entry, acting as a mock
helpdesk) and optionally sends an email alert via SMTP if configured.
"""
import json
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

from src import config


def create_ticket(session_id: str, question: str, reason: str = "user_requested") -> dict:
    os.makedirs(os.path.dirname(config.TICKETS_LOG), exist_ok=True)
    tickets = []
    if os.path.exists(config.TICKETS_LOG):
        with open(config.TICKETS_LOG, "r") as f:
            tickets = json.load(f)

    ticket = {
        "id": len(tickets) + 1,
        "session_id": session_id,
        "question": question,
        "reason": reason,  # "user_requested" or "bot_could_not_help"
        "status": "open",
        "created_at": datetime.utcnow().isoformat(),
    }
    tickets.append(ticket)

    with open(config.TICKETS_LOG, "w") as f:
        json.dump(tickets, f, indent=2)

    if config.ENABLE_EMAIL_HANDOFF:
        _send_email_alert(ticket)

    return ticket


def _send_email_alert(ticket: dict):
    if not (config.SMTP_HOST and config.SMTP_USER and config.SMTP_PASSWORD and config.HANDOFF_TO_EMAIL):
        return  # silently skip if not fully configured

    body = (
        f"New support ticket #{ticket['id']}\n\n"
        f"Session: {ticket['session_id']}\n"
        f"Reason: {ticket['reason']}\n"
        f"Question: {ticket['question']}\n"
        f"Created: {ticket['created_at']}"
    )
    msg = MIMEText(body)
    msg["Subject"] = f"[Support Bot] New ticket #{ticket['id']}"
    msg["From"] = config.SMTP_USER
    msg["To"] = config.HANDOFF_TO_EMAIL

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        # Don't crash the chat flow if email fails — just log it.
        print(f"[handoff] Email alert failed: {e}")


def get_all_tickets() -> list[dict]:
    if not os.path.exists(config.TICKETS_LOG):
        return []
    with open(config.TICKETS_LOG, "r") as f:
        return json.load(f)


def get_unanswered_questions() -> list[dict]:
    if not os.path.exists(config.UNANSWERED_LOG):
        return []
    with open(config.UNANSWERED_LOG, "r") as f:
        return json.load(f)
