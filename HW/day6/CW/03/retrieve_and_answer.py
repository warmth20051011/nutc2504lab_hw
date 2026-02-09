from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)

client = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "cw03_docs"


def retrieve(query, top_k=5):
    vec = embedder.encode(query).tolist()
    res = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=top_k
    )

    if not res.points:
        return []

    return [p.payload["text"] for p in res.points]


def answer(question, contexts):
    prompt = f"""
請根據以下文件內容回答問題，不可編造：

文件內容：
{chr(10).join(contexts)}

問題：
{question}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()

