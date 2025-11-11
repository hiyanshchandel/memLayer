from clients.vector_client import vec_client
import sqlite3
import json
from memory_blob.definition import MemoryBlob
from config import EPISODIC_MEMORY_DB, EPISODIC_COLLECTION_NAME, EPISODIC_THRESHOLD, EPISODIC_TOP_K


class EpisodicMemoryManager:
    def __init__(self, db_path: str = EPISODIC_MEMORY_DB, collection_name: str = EPISODIC_COLLECTION_NAME):
        self.qdrant = vec_client
        self.collection_name = collection_name
        self.db_path = db_path
        
        # SQLite connection
        self.conn = sqlite3.connect(self.db_path)
        self.db = self.conn.cursor()

        # Create table if not exists
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                memory_type TEXT,
                created_at TEXT,
                last_accessed TEXT,
                frequency INTEGER,
                salience REAL,
                version INTEGER,
                tags TEXT
            )
        """)
        self.conn.commit()

    def store_memory(self, memory: MemoryBlob):
        """Insert or update memory in both SQLite and Qdrant."""
        # Store in vector DB
        point = memory.to_vector_point()
        self.qdrant.upsert(collection_name=self.collection_name, points=[point])

        # Store in SQLite (UPSERT)
        self.db.execute("""
            INSERT INTO memories (
                id, content, memory_type, created_at, last_accessed, frequency, salience, version, tags
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content,
                memory_type=excluded.memory_type,
                created_at=excluded.created_at,
                last_accessed=excluded.last_accessed,       
                frequency=excluded.frequency,
                salience=excluded.salience,
                version=excluded.version,
                tags=excluded.tags
        """, (
            memory.id,
            memory.content,
            memory.memory_type,
            memory.created_at,
            memory.last_accessed,
            memory.frequency,
            memory.salience,
            memory.version,
            json.dumps(memory.tags)
        ))

        self.conn.commit()
    
    def retrieve_memory(self, memory_id: str):
        """currently does not fetch from vector DB"""
        """Retrieve a memory by its ID."""
        self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = self.db.fetchone()

        if row:
            memory = {
                "id": row[0],
                "content": row[1],
                "memory_type": row[2],
                "created_at": row[3],
                "last_accessed": row[4],
                "frequency": row[5],
                "salience": row[6],
                "version": row[7],
                "tags": json.loads(row[8])
            }
            return memory
        return None

    def retrieve_similar(self, query_embedding: list[float],
                         limit: int = EPISODIC_TOP_K,
                         threshold: float = EPISODIC_THRESHOLD):
        """Retrieve memories similar to a query embedding."""
        
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using='dense-vector',
            limit=limit,
            score_threshold=threshold
        )

        memories = []
        for point in results.points:
            self.db.execute("SELECT * FROM memories WHERE id = ?", (point.id,))
            row = self.db.fetchone()

            if row:
                memory = {
                    "id": row[0],
                    "content": row[1],
                    "memory_type": row[2],
                    "created_at": row[3],
                    "last_accessed": row[4],
                    "frequency": row[5],
                    "salience": row[6],
                    "version": row[7],
                    "tags": json.loads(row[8]),
                    "similarity": point.score
                }
                memories.append(memory)

        return memories
    


 