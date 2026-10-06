import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


st.set_page_config(page_title="Day 2 - LLM Chains", page_icon="🔗")
st.title("Day 2: LLM Chains")
st.caption("Explore prompt templates, output parsers, and the memory problem.")


@st.cache_resource
def build_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key,
    )


api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "GOOGLE_API_KEY is not configured. Add it under App settings > Secrets "
        "in Streamlit Cloud."
    )
    st.stop()

llm = build_llm(api_key)

tab_explainer, tab_memory = st.tabs(["Topic explainer", "Memory problem"])

with tab_explainer:
    st.subheader("Explain a topic")
    topic = st.text_input("Topic", placeholder="e.g. API")
    if st.button("Explain", type="primary", key="explain"):
        if not topic.strip():
            st.warning("Enter a topic first.")
        else:
            prompt = ChatPromptTemplate.from_template(
                "Explain {topic} in simple terms using an everyday analogy. "
                "Keep it under 3 sentences."
            )
            chain = prompt | llm | StrOutputParser()
            with st.spinner("Generating an explanation..."):
                result = chain.invoke({"topic": topic})
            st.markdown(result)

with tab_memory:
    st.subheader("Ask two independent questions")
    st.caption(
        "Each question is sent independently, so the second answer does not "
        "automatically remember the first."
    )
    question_one = st.text_input("Question 1", key="question_one")
    question_two = st.text_input("Question 2", key="question_two")
    if st.button("Ask both questions", key="ask_both"):
        if not question_one.strip() or not question_two.strip():
            st.warning("Enter both questions first.")
        else:
            prompt = ChatPromptTemplate.from_template(
                "Answer the user's question.\n\nUser: {question}"
            )
            chain = prompt | llm | StrOutputParser()
            with st.spinner("Generating answers..."):
                answers = chain.batch(
                    [{"question": question_one}, {"question": question_two}]
                )
            st.markdown("**Answer 1**")
            st.write(answers[0])
            st.markdown("**Answer 2**")
            st.write(answers[1])
