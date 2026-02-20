# ingestion/email_loader.py
import email
import logging

logger = logging.getLogger(__name__)


def load_email(file_path):
    """Load and extract text body from an .eml email file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            msg = email.message_from_file(f)
    except Exception as e:
        logger.error(f"Failed to read email '{file_path}': {e}")
        return []

    body = ""

    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors="replace")
    except Exception as e:
        logger.error(f"Failed to extract email body from '{file_path}': {e}")
        return []

    if not body.strip():
        logger.warning(f"No text body found in email '{file_path}'")
        return []

    return [{
        "content": body,
        "metadata": {
            "source": file_path,
            "type": "email",
            "subject": msg.get("Subject", ""),
            "from": msg.get("From", ""),
            "date": msg.get("Date", "")
        }
    }]
