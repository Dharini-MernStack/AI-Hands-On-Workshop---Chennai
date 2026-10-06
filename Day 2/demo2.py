import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()

# Create the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,

)
# # # ==========================================
# # # CHAIN 1 — Generate Research Questions
# # # ==========================================

# research_prompt = ChatPromptTemplate.from_template(
#     """
#     You are a research assistant.

#     Given a topic, generate exactly 3 focused
#     research questions that would help someone
#     understand the topic.

#     Topic: {topic}

#     Output ONLY the 3 questions.
#     """
# )

# research_chain = research_prompt | llm | StrOutputParser()
# topic = "Explain how Electric Vehicles (EVs) work and their environmental impact."

# print("\n" + "=" * 60)
# print("CHAIN 1 — GENERATING QUESTIONS")
# print("=" * 60)

# questions = research_chain.invoke({
#     "topic": topic
# })

# print("\nTopic:")
# print(topic)

# print("\nQuestions generated:")
# print(questions)
# # # ==========================================
# # # CHAIN 2 — Answer the Questions
# # # ==========================================

# answer_prompt = ChatPromptTemplate.from_template(
#     """
#     You are an expert researcher.

#     Answer the following research questions
#     concisely.

#     For each question, provide 2-3 key points.

#     Research Questions:
#     {questions}

#     Provide clear and factual answers.
#     """
# )

# answer_chain = answer_prompt | llm | StrOutputParser()
# print("\n" + "=" * 60)
# print("CHAIN 2 — ANSWERING QUESTIONS")
# print("=" * 60)

# research = answer_chain.invoke({
#     "questions": questions
# })

# print("\nResearch:")
# print(research)
# # # ==========================================
# # # CHAIN 3 — Create Executive Summary
# # # ==========================================

# summary_prompt = ChatPromptTemplate.from_template(
#     """
#     You are an executive editor.

#     Based on the research below, create a
#     concise executive summary in 5-6 sentences.

#     Research:
#     {research}

#     Make the summary clear and easy to understand.
#     """
# )

# summary_chain = summary_prompt | llm | StrOutputParser()


# print("\n" + "=" * 60)
# print("CHAIN 3 — CREATING SUMMARY")
# print("=" * 60)

# summary = summary_chain.invoke({
#     "research": research
# })

# print("\nExecutive Summary:")
# print(summary)



# # ==========================================
# # DEMO 3 — MULTI-INPUT CHAIN
# # ==========================================

# lesson_prompt = ChatPromptTemplate.from_template(
#     """
#     Create a 1-hour lesson plan for teaching {topic}
#     to {audience}.

#     Difficulty level: {level}

#     Include:
#     - A real-world hook
#     - Core concept explanation
#     - Hands-on activity
#     - Assessment question

#     Format the answer clearly.
#     """
# )

# lesson_chain = lesson_prompt | llm | StrOutputParser()

# result = lesson_chain.invoke({
#     "topic": "Introduction to APIs",
#     "audience": "college students with basic Python knowledge",
#     "level": "Beginner"
# })

# print("\n" + "=" * 60)
# print("DEMO 3 — MULTI-INPUT CHAIN")
# print("=" * 60)

# print(result)



# # ==========================================
# # DEMO 4 — BATCH PROCESSING
# # ==========================================

# explain_prompt = ChatPromptTemplate.from_template(
#     "Explain {concept} in AI in exactly one simple sentence."
# )

# explain_chain = explain_prompt | llm | StrOutputParser()

# concepts = [
#     {"concept": "hallucination"},
#     {"concept": "fine-tuning"},
#     {"concept": "prompt engineering"},
#     {"concept": "temperature"},
#     {"concept": "tokens"},
# ]

# print("\n" + "=" * 60)
# print("DEMO 4 — BATCH PROCESSING")
# print("=" * 60)

# results = explain_chain.batch(concepts)

# for concept, result in zip(concepts, results):
#     print(f"\n📌 {concept['concept']}")
#     print(result)



# # ==========================================
# # DEMO 5 — THE MEMORY PROBLEM
# # ==========================================

chat_prompt = ChatPromptTemplate.from_template(
    """
    Answer the user's question.

    User: {question}
    """
)

chat_chain = chat_prompt | llm | StrOutputParser()

print("\n" + "=" * 60)
print("DEMO 5 — THE MEMORY PROBLEM")
print("=" * 60)

question1 = input("\nYou: ")
answer1 = chat_chain.invoke({"question": question1})
print("AI:", answer1)

question2 = input("\nYou: ")
answer2 = chat_chain.invoke({"question": question2})
print("AI:", answer2)