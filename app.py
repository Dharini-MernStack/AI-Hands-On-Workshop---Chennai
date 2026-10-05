from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path(__file__).parent / "sample_ai_policy.pdf"


st.set_page_config(page_title="AI Policy Chatbot", page_icon="🤖")
st.title("AI Policy Chatbot")
st.caption("Ask questions about the AI policy document.")


@st.cache_resource
def build_retriever():
    pages = PyPDFLoader(str(PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_chatbot",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


@st.cache_resource
def build_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
    )


def ask_question(question: str, retriever, llm):
    docs = retriever.invoke(question)
    context = "\n\n".join(document.page_content for document in docs)
    prompt = f"""
You are a helpful assistant that answers questions based ONLY on the provided context.

If the answer is not in the context, say:
"I don't have that information in the document."

Context from document:
{context}

Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content, docs


api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "GOOGLE_API_KEY is not configured. Add it under App settings > Secrets "
        "in Streamlit Cloud."
    )
    st.stop()

if not PDF_PATH.exists():
    st.error(f"Could not find the policy document: {PDF_PATH.name}")
    st.stop()

retriever = build_retriever()
llm = build_llm(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

question = st.chat_input("Ask a question about the policy...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy..."):
            answer, documents = ask_question(question, retriever, llm)
        st.markdown(answer)
        sources = [
            document.page_content[:120].replace("\n", " ")
            for document in documents
        ]
        with st.expander("Sources"):
            for source in sources:
                st.write(f"- {source}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
