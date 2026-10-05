from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")
sentences = [
    "I love programming in Python",
    "Python coding is my passion",
    "Chennai has amazing filter coffee",
    "Machine learning is fascinating",
    "AI and deep learning are exciting",
]
embeddings = model.encode(sentences)
print(embeddings.shape)

for i, (sent, emb) in enumerate(zip(sentences, embeddings), 1):
    print(f"Sentence {i}: {sent}")
    print("First 5 numbers:", emb[:5].round(3))


from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

pairs = [
    (0, 1, "Python programming vs Python coding"),
    (0, 2, "Python programming vs Chennai coffee"),
    (3, 4, "Machine learning vs AI/deep learning"),
    (2, 3, "Chennai coffee vs Machine learning"),
]
for i, j, desc in pairs:
    score = cosine_similarity(embeddings[i], embeddings[j])
    print(f"{desc}: {score:.3f}")
import chromadb
client = chromadb.Client()
collection = client.create_collection(
    name="workshop_knowledge",
    metadata={"hnsw:space": "cosine"}
)

print("Collection created:", collection.name)


documents = [
    "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "Machine Learning is a subset of AI that enables systems to learn from data.",
    "Deep Learning uses neural networks with multiple layers to model complex patterns.",
    "Natural Language Processing helps computers understand human language.",
    "Chennai is the capital city of Tamil Nadu and a major tech hub in India.",
    "RAG stands for Retrieval Augmented Generation. It combines search with LLM generation.",
    "Vector databases store data as vectors, enabling similarity search based on meaning.",
]
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)
print("Total documents:", collection.count())

query = "Who is the prime minister of India?"
results = collection.query(
    query_texts=[query],
    n_results=2
)
for doc, distance in zip(
    results['documents'][0],
    results['distances'][0]
):
    print(f"Relevance {1 - distance:.2f}: {doc}")
