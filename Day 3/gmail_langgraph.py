import imaplib
import email
import os

from typing import TypedDict

from email.header import decode_header
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


# ============================================================
# 2. Email helper functions
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
# 3. Read latest email from Gmail
# ============================================================

def get_latest_email():

    print("📬 Reading Gmail...")

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

    result = {
        "sender": decode_text(msg.get("From")),
        "subject": decode_text(msg.get("Subject")),
        "body": get_email_body(msg)
    }

    mail.logout()

    return result

@tool
def search_emails(query: str) -> str:
    """
    Search the user's Gmail inbox for emails matching a query.
    Use this when previous email context is needed.
    """

    print(f"\n🔎 TOOL: search_emails('{query}')")

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
        f'SUBJECT "{query}"'
    )

    email_ids = messages[0].split()

    if not email_ids:
        mail.logout()
        return "No matching emails found."

    results = []

    # Look at the latest 5 matches
    for email_id in email_ids[-5:]:

        status, data = mail.fetch(
            email_id,
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

        results.append(
            f"""
From: {sender}
Subject: {subject}

Body:
{body[:2000]}
"""
        )

    mail.logout()

    return "\n---\n".join(results)

# ============================================================
# 4. LangGraph STATE
# ============================================================

class EmailState(TypedDict):

    sender: str
    subject: str
    body: str

    intent: str
    urgency: str
    needs_reply: bool
    summary: str

    email_context: str

    draft_reply: str

    action: str


# ============================================================
# 5. Gemini
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ============================================================
# 6. NODE 1 — Classify email
# ============================================================

def classify_email(state: EmailState):

    print("\n🧠 NODE: classify_email")

    prompt = f"""
You are an intelligent email assistant.

Analyze this email.

From:
{state["sender"]}

Subject:
{state["subject"]}

Body:
{state["body"]}

Determine:

1. Intent
2. Urgency: Low, Medium, or High
3. Whether a reply is required
4. A short summary

Return exactly in this format:

Intent: ...
Urgency: ...
Needs Reply: Yes or No
Summary: ...
"""

    response = llm.invoke(prompt)

    text = response.content

    print("\nGemini says:")
    print(text)

    # Simple extraction for our first demo

    lines = text.splitlines()

    intent = ""
    urgency = ""
    needs_reply = False
    summary = ""

    for line in lines:

        lower = line.lower()

        if lower.startswith("intent:"):
            intent = line.split(":", 1)[1].strip()

        elif lower.startswith("urgency:"):
            urgency = line.split(":", 1)[1].strip()

        elif lower.startswith("needs reply:"):

            value = line.split(
                ":",
                1
            )[1].strip().lower()

            needs_reply = value == "yes"

        elif lower.startswith("summary:"):
            summary = line.split(
                ":",
                1
            )[1].strip()

    return {
        "intent": intent,
        "urgency": urgency,
        "needs_reply": needs_reply,
        "summary": summary
    }


# ============================================================
# 7. NODE 2 — Decide what to do
# ============================================================

def decide_action(state: EmailState):

    print("\n🔀 NODE: decide_action")

    if state["urgency"].lower() == "high":

        action = "flag_for_review"

    elif state["needs_reply"]:

        action = "draft_reply"

    else:

        action = "no_action"

    print(f"Decision: {action}")

    return {
        "action": action
    }

# ============================================================
# 8. NODE 3 — Draft reply
# ============================================================

def draft_reply(state: EmailState):

    print("\n✍️ NODE: draft_reply")

    prompt = f"""
You are helping me reply to an email.

Original email:

From:
{state["sender"]}

Subject:
{state["subject"]}

Body:
{state["body"]}

Write a concise, professional reply.

Rules:

- Do not invent facts.
- Do not make commitments that are not present.
- Be polite.
- Keep it short.
- Return only the email body.
"""

    response = llm.invoke(prompt)

    draft = response.content

    print("\nDraft generated:")
    print("-" * 50)
    print(draft)
    print("-" * 50)

    return {
        "draft_reply": draft
    }

# ============================================================
# 9. NODE 4 — Flag important email for human review
# ============================================================

def flag_for_review(state: EmailState):

    print("\n🚨 NODE: flag_for_review")

    print("\n⚠️ IMPORTANT EMAIL")
    print("Subject:", state["subject"])
    print("Urgency:", state["urgency"])
    print("Summary:", state["summary"])

    return {}
# ============================================================
# 9. CONDITIONAL ROUTING
# ============================================================

def route_after_decision(state: EmailState):

    if state["action"] == "draft_reply":

        return "draft_reply"

    elif state["action"] == "flag_for_review":

        return "flag_for_review"

    return "end"

# ============================================================
# 10. BUILD GRAPH
# ============================================================

builder = StateGraph(EmailState)


builder = StateGraph(EmailState)


builder.add_node(
    "classify",
    classify_email
)

builder.add_node(
    "decide",
    decide_action
)

builder.add_node(
    "draft_reply",
    draft_reply
)

builder.add_node(
    "flag_for_review",
    flag_for_review
)

builder.add_edge(
    START,
    "classify"
)

builder.add_edge(
    "classify",
    "decide"
)


builder.add_conditional_edges(
    "decide",
    route_after_decision,
    {
        "draft_reply": "draft_reply",
        "flag_for_review": "flag_for_review",
        "end": END
    }
)

builder.add_edge(
    "draft_reply",
    END
)
builder.add_edge(
    "flag_for_review",
    END
)


graph = builder.compile()


# ============================================================
# 11. GET EMAIL
# ============================================================

email_data = get_latest_email()


# ============================================================
# 12. INITIAL STATE
# ============================================================

initial_state = {

    "sender": email_data["sender"],

    "subject": email_data["subject"],

    "body": email_data["body"],

    "intent": "",

    "urgency": "",

    "email_context": "",

    "needs_reply": False,

    "summary": "",

    "draft_reply": "",

    "action": ""
}


# ============================================================
# 13. RUN LANGGRAPH
# ============================================================

print("\n🚀 Starting LangGraph...\n")

result = graph.invoke(
    initial_state
)


# ============================================================
# 14. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)

print("FINAL RESULT")

print("=" * 60)

print("Intent:", result["intent"])

print("Urgency:", result["urgency"])

print("Needs reply:", result["needs_reply"])

print("Action:", result["action"])

if result["draft_reply"]:

    print("\nDraft:")
    print(result["draft_reply"])

print("=" * 60)


print(
    search_emails.invoke(
        "meeting"
    )
)