import imaplib
import email
import os

from email.header import decode_header
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


# ============================================================
# 2. Helper: decode email headers
# ============================================================

def decode_text(value):

    if not value:
        return ""

    decoded = decode_header(value)

    result = ""

    for part, encoding in decoded:

        if isinstance(part, bytes):
            result += part.decode(
                encoding or "utf-8",
                errors="ignore"
            )
        else:
            result += part

    return result


# ============================================================
# 3. Helper: extract email body
# ============================================================

def get_email_body(msg):

    if msg.is_multipart():

        for part in msg.walk():

            content_type = part.get_content_type()

            if content_type == "text/plain":

                payload = part.get_payload(
                    decode=True
                )

                if payload:

                    return payload.decode(
                        "utf-8",
                        errors="ignore"
                    )

    else:

        payload = msg.get_payload(
            decode=True
        )

        if payload:

            return payload.decode(
                "utf-8",
                errors="ignore"
            )

    return ""


# ============================================================
# 4. Read latest email from Gmail
# ============================================================

print("Connecting to Gmail...")

mail = imaplib.IMAP4_SSL(
    "imap.gmail.com",
    993
)

mail.login(
    GMAIL_ADDRESS,
    GMAIL_APP_PASSWORD
)

mail.select("INBOX")

status, messages = mail.search(
    None,
    "ALL"
)

email_ids = messages[0].split()

latest_email_id = email_ids[-1]

status, data = mail.fetch(
    latest_email_id,
    "(RFC822)"
)

raw_email = data[0][1]

msg = email.message_from_bytes(
    raw_email
)

sender = decode_text(
    msg.get("From")
)

subject = decode_text(
    msg.get("Subject")
)

body = get_email_body(msg)

mail.logout()


# ============================================================
# 5. Create Gemini model
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# 6. Ask Gemini to understand the email
# ============================================================

prompt = f"""
You are an email assistant.

Analyze the following email.

EMAIL
-----
From: {sender}
Subject: {subject}

Body:
{body}
-----

Return:

1. Intent
2. Urgency: Low, Medium, or High
3. Whether a reply is needed
4. A short summary
5. Reason for your decision

Keep the answer concise.
"""

print("\n🤖 Asking Gemini to understand the email...\n")

response = llm.invoke(prompt)

print("=" * 60)
print("GEMINI ANALYSIS")
print("=" * 60)

print(response.content)

print("=" * 60)