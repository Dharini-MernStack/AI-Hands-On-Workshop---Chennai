import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


st.set_page_config(page_title="Day 3 - Gmail Agents", page_icon="📧")
st.title("Day 3: Gmail Agents")
st.caption("Explore tools, email classification, and agent-style workflows.")


@st.cache_resource
def build_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )


@tool
def search_workshop_emails(query: str) -> list[str]:
    """Search a sample inbox for messages matching a keyword."""
    emails = [
        "Workshop schedule for Chennai",
        "Reminder: Mentor workshop tomorrow",
        "Student attendance report",
        "AI workshop resources",
        "Team meeting agenda",
    ]
    return [message for message in emails if query.lower() in message.lower()]


api_key = st.secrets.get("GOOGLE_API_KEY")

tab_tools, tab_classifier, tab_reference = st.tabs(
    ["Tools", "Email classifier", "Day 3 files"]
)

with tab_tools:
    st.subheader("Normal function vs. tool")
    st.write(
        "A normal function runs only when your code calls it. A tool gives an "
        "agent a description and schema it can use to decide when to call it."
    )
    query = st.text_input("Search the sample inbox", placeholder="e.g. workshop")
    if st.button("Search emails", type="primary"):
        if not query.strip():
            st.warning("Enter a search term first.")
        else:
            results = search_workshop_emails.invoke({"query": query})
            if results:
                for result in results:
                    st.write(f"- {result}")
            else:
                st.info("No matching messages found.")

with tab_classifier:
    st.subheader("Classify an email with Gemini")
    if not api_key:
        st.error(
            "GOOGLE_API_KEY is not configured. Add it under App settings > "
            "Secrets in Streamlit Cloud."
        )
    else:
        sender = st.text_input("Sender", placeholder="mentor@example.com")
        subject = st.text_input("Subject", placeholder="Workshop reminder")
        body = st.text_area(
            "Email body",
            height=180,
            placeholder="Paste the email text here...",
        )
        if st.button("Analyze email", type="primary"):
            if not body.strip():
                st.warning("Paste an email body first.")
            else:
                prompt = ChatPromptTemplate.from_template(
                    """You are an email assistant. Analyze the email below.

Return:
1. Intent
2. Urgency: Low, Medium, or High
3. Whether a reply is needed
4. A short summary
5. Reason for your decision

From: {sender}
Subject: {subject}
Body:
{body}

Keep the answer concise."""
                )
                chain = prompt | build_llm(api_key) | StrOutputParser()
                with st.spinner("Analyzing the email..."):
                    result = chain.invoke(
                        {"sender": sender, "subject": subject, "body": body}
                    )
                st.markdown(result)

with tab_reference:
    st.subheader("Standalone workshop demos")
    st.write(
        "The complete Day 3 Python demonstrations remain in the repository "
        "under the `Day 3` folder."
    )
    st.code(
        "fix_memory.py\ngmail_agent_test.py\ngmail_classify.py\n"
        "gmail_langgraph.py\ngmail_test.py\nnf_vs_tool.py",
        language="text",
    )
