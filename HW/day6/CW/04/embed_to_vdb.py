import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

DATA_DIR = "."
COLLECTION = "cw04_hybrid_docs"

CHUNK_SIZE = 500
OVERLAP = 50

QDRANT_URL = "http://localhost:6333"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_URL)


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
                    docs.append({
                        "text": chunk,
                        "source": fn
                    })

    print(f"Total chunks: {len(docs)}")

    embeddings = embedder.encode([d["text"] for d in docs]).tolist()
    dim = len(embeddings[0])

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )

    points = [
        models.PointStruct(
            id=i,
            vector={"dense": embeddings[i]},
            payload={
                "text": docs[i]["text"],
                "source": docs[i]["source"]
            }
        )
        for i in range(len(docs))
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print("✅ Hybrid collection 建立完成")


if __name__ == "__main__":
    main()

