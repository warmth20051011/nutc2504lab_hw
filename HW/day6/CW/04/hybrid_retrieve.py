from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "cw04_hybrid_docs"

client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def hybrid_retrieve(query: str, top_k: int = 5, top_n: int = 3):
    """
    Dense + Sparse (BM25) Hybrid Search with RRF
    """

    # Dense embedding
    query_vec = embedder.encode(query).tolist()

    result = client.query_points(
        collection_name=COLLECTION,

        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25"
                ),
                using="sparse",
                limit=top_k
            ),
            models.Prefetch(
                query=query_vec,
                using="dense",
                limit=top_k
            ),
        ],


        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),

        limit=top_n,
        with_payload=True
    )

    return [p.payload["text"] for p in result.points]

