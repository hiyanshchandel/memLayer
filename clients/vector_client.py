from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

vec_client = QdrantClient(
    host = "localhost",
    port = 6333,
)   