from langchain_core.tools import tool

# ============================================================
# DEMO 1 — NORMAL PYTHON FUNCTION
# ============================================================

emails = [
    "Workshop schedule for Chennai",
    "Reminder: Mentor workshop tomorrow",
    "Student attendance report",
    "AI workshop resources",
    "Team meeting agenda"
]


def search_emails(query):
    """Normal Python function."""
    
    results = [
        email for email in emails
        if query.lower() in email.lower()
    ]
    
    return results


# WE explicitly decide to call the function
print("NORMAL FUNCTION:")
print(search_emails("reminder"))


# ============================================================
# DEMO 2 — TURN THE FUNCTION INTO A TOOL
# ============================================================

@tool
def search_emails_tool(query: str):
    """
    Search the user's email inbox for messages
    matching the given keyword.
    """
    
    results = [
        email for email in emails
        if query.lower() in email.lower()
    ]
    
    return results


print("\nTOOL:")
print(search_emails_tool.invoke({"query": "workshop"}))