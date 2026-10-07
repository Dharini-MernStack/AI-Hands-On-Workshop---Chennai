import imaplib
import email
import os

from pyexpat.errors import messages
from typing import Annotated, TypedDict

from email.header import decode_header
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.tools import tool
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage
)

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


# ============================================================
# 2. EMAIL HELPERS
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

            if part.get_content_type() == "text/plain":

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
# 3. TOOL — SEARCH EMAILS
# ============================================================

@tool
def search_emails(query: str) -> str:
    """
    Search the user's Gmail inbox for previous emails
    matching a keyword or phrase.

    Use this tool when the current email refers to
    previous conversations or information that is not
    available in the current email.
    """

    print(
        f"\n🔎 TOOL CALLED: search_emails('{query}')"
    )

    mail = imaplib.IMAP4_SSL(
        "imap.gmail.com",
        993
    )

    mail.login(
        GMAIL_ADDRESS,
        GMAIL_APP_PASSWORD
    )

    mail.select("INBOX")

    # Search both subject and body text
    status, messages = mail.search(
    None,
    f'SUBJECT "{query}"'
)

    email_ids = messages[0].split()

    if not email_ids:

        mail.logout()

        return "No matching emails found."

    results = []

    # Take latest 5 matching emails
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

    return "\n--------------------\n".join(results)

# ============================================================
# TOOL 2 — GET EMAIL THREAD / CONVERSATION
# ============================================================

@tool
def get_email_thread(subject: str) -> str:
    """
    Retrieve recent emails from the inbox that belong to
    the same conversation or have the same subject.

    Use this when you need more context from a previous
    email conversation.
    """

    print(
        f"\n🧵 TOOL CALLED: get_email_thread('{subject}')"
    )

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
        f'SUBJECT "{subject}"'
    )

    email_ids = messages[0].split()

    if not email_ids:

        mail.logout()

        return "No previous conversation found."

    results = []

    # Get the latest 10 matching emails
    for email_id in email_ids[-10:]:

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

        date = decode_text(
            msg.get("Date")
        )

        body = get_email_body(msg)

        results.append(
            f"""
From: {sender}
Date: {date}
Subject: {decode_text(msg.get("Subject"))}

Body:
{body[:3000]}
"""
        )

    mail.logout()

    return "\n====================\n".join(
        results
    )


@tool
def draft_reply(
    recipient: str,
    subject: str,
    context: str
) -> str:
    """
    Draft a professional email reply based on the
    provided context.

    Use this when the user wants a reply drafted.
    This tool only creates a draft. It does not send email.
    """

    print(f"\n✍️ TOOL CALLED: draft_reply('{subject}')")

    prompt = f"""
    Draft a professional and concise email reply.

    Recipient:
    {recipient}

    Subject:
    {subject}

    Context:
    {context}

    Requirements:
    - Be professional and friendly.
    - Do not invent facts.
    - Use only the information provided in the context.
    - Keep the reply concise.
    - Do not include a subject line.
    - Return only the email body.
    """

    response = llm.invoke(prompt)

    return response.content
# ============================================================
# 4. REGISTER TOOLS
# ============================================================

tools = [
    search_emails,
    get_email_thread,
    draft_reply
]

# ============================================================
# 5. GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# Give Gemini access to our tools
llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 6. AGENT STATE
# ============================================================

class AgentState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]


# ============================================================
# 7. AGENT NODE
# ============================================================

def agent(state: AgentState):

    print("\n🤖 NODE: AGENT")

    system_message = SystemMessage(
    content="""
You are an intelligent personal email assistant.

You have access to three tools:

1. search_emails
   Use this to find relevant previous emails.

2. get_email_thread
   Use this when you need conversation history
   for a particular email subject.

3. draft_reply
   Use this when the user wants a reply drafted.
   This tool only creates a draft. It does NOT send email.

Your workflow is flexible.

Use tools only when necessary.

If previous context is needed:

1. Search for relevant emails.
2. Retrieve the conversation history if necessary.
3. Use the information you found to understand the situation.
4. If the user asks for a reply, use draft_reply.

Do not invent information.

Do not send emails.

Once you have enough information, provide the answer.
"""
)
    messages = [
        system_message
    ] + state["messages"]

    response = llm_with_tools.invoke(
        messages
    )

    # Show what the agent decided to do
    if response.tool_calls:

        print("\n🧠 Agent decided to use a tool:")

        for call in response.tool_calls:

            print(
                "Tool:",
                call["name"]
            )

            print(
                "Arguments:",
                call["args"]
            )

    else:

        print("\n✅ Agent decided no tool was needed.")

        print(
            "\nAgent response:"
        )

        print(
            response.content
        )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# 8. TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# 9. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(
    AgentState
)

builder.add_node(
    "agent",
    agent
)

builder.add_node(
    "tools",
    tool_node
)


# START → AGENT

builder.add_edge(
    START,
    "agent"
)


# AGENT → TOOLS or END

builder.add_conditional_edges(
    "agent",
    tools_condition
)


# TOOLS → AGENT

builder.add_edge(
    "tools",
    "agent"
)


graph = builder.compile()


# ============================================================
# 10. TEST THE AGENT
# ============================================================

test_message = """
I need to reply to the subject line that calls out 'Pigeon &amp; Safety Net All towers Final Invoice generated for T1 A-806'.

Please:

1. Find the relevant previous emails from the sender if any
2. Retrieve the conversation history.
3. Understand what was the mail about
4. Draft a professional reply thanking him.
Do not send the email.
"""

print("\n🚀 STARTING AGENT...\n")


result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content=test_message
            )
        ]
    }
)


# ============================================================
# 11. FINAL RESPONSE
# ============================================================

print("\n" + "=" * 60)

print("FINAL AGENT RESPONSE")

print("=" * 60)

final_response = result["messages"][-1].content

print("\n" + "=" * 60)
print("FINAL AGENT RESPONSE")
print("=" * 60)

if isinstance(final_response, list):

    for item in final_response:

        if isinstance(item, dict) and item.get("type") == "text":
            print(item["text"])

else:
    print(final_response)