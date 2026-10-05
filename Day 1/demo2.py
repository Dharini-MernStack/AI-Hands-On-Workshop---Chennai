from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("sample_ai_policy.pdf")
pages = loader.load()
print("Pages loaded:", len(pages))
print(pages[0].page_content[:200])


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=5
)
chunks = splitter.split_documents(pages)
print("Number of chunks:", len(chunks))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf_chatbot"
)
print("Vector store created with", len(chunks), "chunks")

query = "What AI tools can employees use?"
results = vectorstore.similarity_search(
    query,
    k=2
)
for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(doc.page_content)

    from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

response = llm.invoke("Say hello in Tamil!")
print(response.content)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# RAG function

def ask(question):

    # 1. RETRIEVE
    docs = retriever.invoke(question)

    context = "\n\n".join(
        d.page_content for d in docs
    )

    # 2. AUGMENT
    prompt = f"""
You are a helpful assistant that answers questions
based ONLY on the provided context.

If the answer is not in the context, say:
"I don't have that information in the document."

Context from document:
{context}

Question:
{question}

Answer:
"""

    # 3. GENERATE
    response = llm.invoke(prompt)

    return response.content, docs


print("\nChat with your PDF! Type 'quit' to exit.\n")

while True:

    question = input("You: ").strip()

    if question.lower() in ["quit", "exit", "q"]:
        break

    if not question:
        continue

    answer, docs = ask(question)

    print(f"Bot: {answer}\n")

    print("Sources:")

    for d in docs:
        print(
            " -",
            d.page_content[:80].replace("\n", " ")
        )

    print()
