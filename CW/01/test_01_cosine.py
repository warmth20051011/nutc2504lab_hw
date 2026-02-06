from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="CW01_cosine",
    vectors_config=VectorParams(
        size=4096,
        distance=Distance.COSINE
    )
)

print("Step 1 COSINE collection created")

