from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import os
from dotenv import load_dotenv
from embeddings import get_embedding
from datetime import datetime
import uuid
from summarize import summarize_text

client = QdrantClient(
    host = "localhost",
    port = 6333,
)

def create_points(history):
    history = [f"{role}: {text}" for role, text in history]

    summarised = summarize_text(history)

    ai_insights = summarised.get("ai_insights", [])
    user_insights = summarised.get("user_insights", [])

    if not ai_insights and not user_insights:
        return []
    
    points = []
    
    # Add AI insights
    for insight in ai_insights:
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense-vector": get_embedding(insight)},  
            payload={
                "role": "AI",
                "text": insight,
                "timestamp": datetime.now().isoformat()
            }
        ))
    
    # Add User insights
    for insight in user_insights:
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense-vector": get_embedding(insight)},  
            payload={
                "role": "User",
                "text": insight,
                "timestamp": datetime.now().isoformat()
            }
        ))
    
    return points


def upsert_vector_batch(collection_name: str, history):
    points = create_points(history[-4:])
    if not points:
        print("No points to upsert.")
        return
    
    client.upsert(collection_name=collection_name, points=points)