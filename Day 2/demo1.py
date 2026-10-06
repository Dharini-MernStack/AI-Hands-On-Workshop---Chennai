import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env
load_dotenv()

# 1. Create the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

# 2. Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms using an everyday analogy. "
    "Keep it under 3 sentences."
)

# 3. Connect Prompt → LLM → Output Parser
chain = prompt | llm | StrOutputParser()

# 4. Run the chain
# result = chain.invoke({
#     "topic": "API"
# })
while True:
    topic = input("\nEnter a topic (or type 'exit'): ")

    if topic.lower() == "exit":
        break

    result = chain.invoke({
        "topic": topic
    })

    print("\n🤖 Gemini says:")
    print(result)


# # 5. Print the answer
# print("\n🤖 Gemini says:")
# print(result)
