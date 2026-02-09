import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

DATA_DIR = "."
COLLECTION = "cw04_hybrid_docs"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")


def load_docs():
    docs = []
    for fn in os.listdir(DATA_DIR):
        if fn.startswith("data_") and fn.endswith(".txt"):
            with open(fn, encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def main():
    docs = load_docs()
    embeddings = model.encode(docs).tolist()
    dim = len(embeddings[0])

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    points = []
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        points.append(
            models.PointStruct(
                id=i,
                vector={
                    "dense": emb,
                    "sparse": models.Document(
                        text=doc,
                        model="Qdrant/bm25"
                    )
                },
                payload={"text": doc}
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)
    print(" CW04 Hybrid collection 建立完成")


if __name__ == "__main__":
    main()

