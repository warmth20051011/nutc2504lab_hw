import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

DATA_DIR = "."
COLLECTION = "cw03_docs"

CHUNK_SIZE = 500
OVERLAP = 50

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")


def sliding_chunk(text):
    step = CHUNK_SIZE - OVERLAP
    return [
        text[i:i + CHUNK_SIZE]
        for i in range(0, len(text), step)
        if len(text[i:i + CHUNK_SIZE]) > 100
    ]


def main():
    docs = []

    for fn in os.listdir(DATA_DIR):
        if fn.startswith("data_") and fn.endswith(".txt"):
            with open(fn, encoding="utf-8") as f:
                for chunk in sliding_chunk(f.read()):
                    docs.append({"text": chunk, "source": fn})

    embeddings = model.encode([d["text"] for d in docs]).tolist()

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=len(embeddings[0]),
            distance=Distance.COSINE
        )
    )

    points = [
        PointStruct(id=i, vector=embeddings[i], payload=docs[i])
        for i in range(len(docs))
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print("✅ Sliding Window chunks 已嵌入 Qdrant")


if __name__ == "__main__":
    main()

