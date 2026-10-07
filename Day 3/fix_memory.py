"""
🔷 DAY 3 - DEMO 5a: Memory & Tools
=========================================================

Goal:
1. Understand the "goldfish" problem
2. Build memory using conversation history
3. Introduce tools
4. Show the difference between a normal function and an agent tool

"""

import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage


# ============================================
# LLM SETUP
# ============================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# ============================================
# PART 1: WITHOUT MEMORY
# ============================================

# print("=" * 60)
# print("PART 1: WITHOUT Memory (Goldfish Mode 🐟)")
# print("=" * 60)

# print("""
# The LLM does NOT automatically remember previous calls.

# Every time we call:

#     llm.invoke(...)

# we are essentially starting a new conversation.
# """)

# print("\nMessage 1:")

# message_1 = HumanMessage(
#     content="My name is Dharini and I teach Python in Chennai."
# )

# response_1 = llm.invoke([message_1])

# print(f"🤖: {response_1.content[:200]}")


# print("\nMessage 2:")

# message_2 = HumanMessage(
#     content="What's my name and what do I teach?"
# )

# response_2 = llm.invoke([message_2])

# print(f"🤖: {response_2.content[:300]}")

# print("\n❌ The LLM doesn't know because we didn't give it Message 1.")


# ============================================
# PART 1B: WITH MEMORY
# ============================================

print("\n" + "=" * 60)
print("PART 1B: WITH Memory (Elephant Mode 🐘)")
print("=" * 60)


# Our memory
conversation_history = []


def chat_with_memory(user_message):
    """
    Add the user's message to memory,
    send the complete conversation to the LLM,
    then store the AI response.
    """

    # 1. Store user message
    conversation_history.append(
        HumanMessage(content=user_message)
    )

    # 2. Send entire conversation to LLM
    response = llm.invoke(conversation_history)

    # 3. Store AI response
    conversation_history.append(
        AIMessage(content=response.content)
    )

    return response.content


print("\nMessage 1:")

r1 = chat_with_memory(
    "My name is Dharini and I teach Python in Chennai."
)

print(f"🤖: {r1[:200]}")


print("\nMessage 2:")

r2 = chat_with_memory(
    "What's my name and what do I teach?"
)

print(f"🤖: {r2[:300]}")

print("\n✅ It remembers because we sent the previous conversation.")


print("\nMessage 3:")

r3 = chat_with_memory(
    "Suggest a fun Python project for my students."
)

print(f"🤖: {r3[:300]}")

print("\n✅ It can use the previous context as well.")


# ============================================
# LOOK INSIDE MEMORY
# ============================================

print("\n" + "=" * 60)
print("📦 WHAT IS ACTUALLY STORED IN MEMORY?")
print("=" * 60)

for message in conversation_history:

    if isinstance(message, HumanMessage):
        role = "Human"

    elif isinstance(message, AIMessage):
        role = "AI"

    else:
        role = "Unknown"

    print(f"\n[{role}]")
    print(message.content[:200])


# ============================================
# IMPORTANT CONCEPT
# ============================================

print("\n" + "=" * 60)
print("🔑 KEY MEMORY INSIGHT")
print("=" * 60)

print("""
The LLM itself is NOT storing the memory.

Our Python program stores:

    conversation_history

Every new question is sent together with
the previous conversation.

So:

    User
      ↓
    Memory
      ↓
    Previous conversation + New question
      ↓
    LLM
      ↓
    Response
      ↓
    Memory
""")