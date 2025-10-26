import uuid
import time
from embeddings import get_embedding

class MemoryBlob:
    def __init__(self, content: str, memory_type: str = "episodic", embedding: list[float] = None, id: str = None, created_at: str = None, tags: dict = None):
        self.id = id or str(uuid.uuid4())
        self.memory_type = memory_type
        self.content = content
        self.embedding = embedding
        self.created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.tags = tags or {}
        self.frequency = 1
        self.last_accessed = self.created_at
        self.salience = 0.0
        self.version = 1

    def create_embedding(self):
        if self.embedding is None:
            self.embedding = get_embedding(self.content)
        return self.embedding
    
    def to_vector_point(self):
        return {
            "id": self.id,
            "vector": self.create_embedding(), 
            "payload": {
                "content": self.content,
                "memory_type": self.memory_type,
                "created_at": self.created_at,
                "tags": self.tags,
                "frequency": self.frequency,
                "last_accessed": self.last_accessed,
                "salience": self.salience,
                "version": self.version
            },
        }

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "memory_type": self.memory_type,
            "created_at": self.created_at,
            "tags": self.tags,
            "frequency": self.frequency,
            "last_accessed": self.last_accessed,
            "salience": self.salience,
            "version": self.version
        }
    
