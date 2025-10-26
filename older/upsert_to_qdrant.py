from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import os
from dotenv import load_dotenv
from embeddings import get_embedding
from datetime import datetime
import uuid
from .summarize import summarize_text

client = QdrantClient(
    host = "localhost",
    port = 6333,
)

def get_insights(history) -> list:
    history = [f"{role}: {text}" for role, text in history]
    summarised = summarize_text(history)
    ai_insights = summarised.get("ai_insights", [])
    user_insights = summarised.get("user_insights", [])
    print(f"Extracted {len(ai_insights)} AI insights and {len(user_insights)} User insights.")
    return [ai_insights, user_insights]


def upsert_vector(collection_name: str, role: str, text: str, point_id: str = None):
    """Upsert a vector into Qdrant."""
    if point_id is None:
        point_id = str(uuid.uuid4())

    point = PointStruct(
        id=point_id,
        vector={"dense-vector": get_embedding(text)},  
        payload={
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
    )
    client.upsert(collection_name=collection_name, points=[point])