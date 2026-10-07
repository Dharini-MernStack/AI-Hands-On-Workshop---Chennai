# import imaplib
# import os
# from dotenv import load_dotenv
# load_dotenv()

# GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
# GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# print("Connecting to Gmail...")

# mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)

# mail.login(
#     GMAIL_ADDRESS,
#     GMAIL_APP_PASSWORD
# )

# print("✅ Gmail login successful!")

# mail.select("INBOX")

# status, messages = mail.search(None, "ALL")

# email_ids = messages[0].split()

# print(f"📬 Emails in inbox: {len(email_ids)}")

# mail.logout()

import imaplib
import email
import os

from email.header import decode_header
from dotenv import load_dotenv

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


def decode_text(value):
    if not value:
        return ""

    decoded = decode_header(value)

    result = ""

    for part, encoding in decoded:
        if isinstance(part, bytes):
            result += part.decode(encoding or "utf-8", errors="ignore")
        else:
            result += part

    return result


def get_email_body(msg):
    """Extract readable text from an email."""

    if msg.is_multipart():

        for part in msg.walk():

            content_type = part.get_content_type()
            content_disposition = str(
                part.get("Content-Disposition", "")
            )

            if (
                content_type == "text/plain"
                and "attachment" not in content_disposition
            ):
                payload = part.get_payload(decode=True)

                if payload:
                    return payload.decode(
                        "utf-8",
                        errors="ignore"
                    )

    else:

        payload = msg.get_payload(decode=True)

        if payload:
            return payload.decode(
                "utf-8",
                errors="ignore"
            )

    return ""


# ----------------------------------------
# Connect to Gmail
# ----------------------------------------

print("Connecting to Gmail...")

mail = imaplib.IMAP4_SSL(
    "imap.gmail.com",
    993
)

mail.login(
    GMAIL_ADDRESS,
    GMAIL_APP_PASSWORD
)

print("✅ Gmail login successful")


# ----------------------------------------
# Open Inbox
# ----------------------------------------

mail.select("INBOX")


# ----------------------------------------
# Get latest emails
# ----------------------------------------

status, messages = mail.search(
    None,
    "ALL"
)

email_ids = messages[0].split()

print(f"📬 Emails found: {len(email_ids)}")


# ----------------------------------------
# Pick latest email
# ----------------------------------------

latest_email_id = email_ids[-1]


status, data = mail.fetch(
    latest_email_id,
    "(RFC822)"
)


raw_email = data[0][1]

msg = email.message_from_bytes(
    raw_email
)


# ----------------------------------------
# Extract information
# ----------------------------------------

sender = decode_text(msg.get("From"))
subject = decode_text(msg.get("Subject"))
body = get_email_body(msg)


print("\n" + "=" * 60)

print("FROM:")
print(sender)

print("\nSUBJECT:")
print(subject)

print("\nBODY:")
print(body)

print("=" * 60)


mail.logout()

print("\n✅ Done")