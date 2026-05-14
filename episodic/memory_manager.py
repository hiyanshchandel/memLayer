from clients.vector_client import vec_client
from clients.openai_client import openai_client
import sqlite3
import json
import re
import time
from datetime import datetime
from memory_blob.definition import MemoryBlob
from config import (
    CHAT_MEMORY_TYPES,
    DOCUMENT_MEMORY_TYPES,
    EPISODIC_MEMORY_DB,
    EPISODIC_COLLECTION_NAME,
    EPISODIC_THRESHOLD,
    EPISODIC_MERGE_THRESHOLD,
    EPISODIC_TOP_K,
    semantic_extraction_model,
)


class EpisodicMemoryManager:
    def __init__(self, db_path: str = EPISODIC_MEMORY_DB, collection_name: str = EPISODIC_COLLECTION_NAME):
        self.qdrant = vec_client
        self.openai_client = openai_client
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
        """Insert or update a memory using deterministic content-only deduplication."""
        resolution = self.resolve_memory_for_storage(memory)
        if resolution["action"] != "duplicate":
            self.persist_memory(resolution["memory_blob"])
        return resolution

    def persist_memory(self, memory: MemoryBlob):
        """Persist the memory to Qdrant and SQLite."""
        return self._upsert_memory_record(memory)

    def _upsert_memory_record(self, memory: MemoryBlob):
        """Persist the memory to Qdrant and SQLite."""
        start = time.perf_counter()
        point = memory.to_vector_point()
        self.qdrant.upsert(collection_name=self.collection_name, points=[point])

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
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[Timing] episodic_upsert_ms={elapsed_ms:.1f}")

    def _merge_content(self, existing_content: str, new_content: str) -> str:
        """Merge two content strings while keeping only unique non-empty lines."""
        merged_lines = []
        seen_lines = set()

        for block in (existing_content, new_content):
            for line in str(block).splitlines():
                normalized_line = line.strip()
                if not normalized_line:
                    continue
                if normalized_line not in seen_lines:
                    seen_lines.add(normalized_line)
                    merged_lines.append(normalized_line)

        return "\n".join(merged_lines)

    def _memory_blob_from_row(self, row):
        memory = MemoryBlob(
            content=row[1],
            memory_type=row[2],
            id=row[0],
            created_at=row[3],
            tags=json.loads(row[8]),
        )
        memory.last_accessed = row[4]
        memory.frequency = row[5]
        memory.salience = row[6]
        memory.version = row[7]
        return memory

    def _resolve_dedup_policy(self, memory: MemoryBlob) -> str:
        memory_type = (memory.memory_type or "").strip().lower()
        source_type = str((memory.tags or {}).get("source_type", "")).strip().lower()

        if memory_type in CHAT_MEMORY_TYPES or source_type in CHAT_MEMORY_TYPES:
            return "chat"

        if memory_type in DOCUMENT_MEMORY_TYPES or source_type in DOCUMENT_MEMORY_TYPES:
            return "document"

        return "document"

    def _find_exact_duplicate(self, memory: MemoryBlob):
        self.db.execute(
            "SELECT * FROM memories WHERE content = ? AND memory_type = ? ORDER BY last_accessed DESC LIMIT 1",
            (memory.content, memory.memory_type),
        )
        row = self.db.fetchone()

        if row:
            return self._memory_blob_from_row(row)

        return None

    def _merge_content_with_llm(self, existing_content: str, new_content: str) -> str:
        """Use an LLM to rewrite the older memory with the new information."""
        start = time.perf_counter()
        response = self.openai_client.chat.completions.create(
            model=semantic_extraction_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You merge two episodic memories into one updated memory. "
                        "Keep all unique facts, remove repetition, and write a concise "
                        "single memory in plain language. Return only valid JSON with "
                        'this schema: {"merged_content": "..."}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Old memory:\n{existing_content}\n\n"
                        f"New memory:\n{new_content}\n\n"
                        "Write the updated merged memory now."
                    ),
                },
            ],
        )

        raw_content = response.choices[0].message.content or ""
        cleaned_content = re.sub(r"^```json\s*|\s*```$", "", raw_content.strip())
        parsed_content = json.loads(cleaned_content)
        merged_content = parsed_content.get("merged_content", "").strip()
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[Timing] episodic_llm_merge_ms={elapsed_ms:.1f}")

        if not merged_content:
            return self._merge_content(existing_content, new_content)

        return merged_content

    def find_similar_memories(self, content: str, query_embedding: list[float] | None = None,
                              limit: int = EPISODIC_TOP_K,
                              threshold: float = EPISODIC_THRESHOLD):
        """Find similar memories using content only."""
        start = time.perf_counter()
        if query_embedding is None:
            query_embedding = MemoryBlob(content=content).create_embedding()
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
                memories.append({
                    "id": row[0],
                    "content": row[1],
                    "memory_type": row[2],
                    "created_at": row[3],
                    "last_accessed": row[4],
                    "frequency": row[5],
                    "salience": row[6],
                    "version": row[7],
                    "tags": json.loads(row[8]),
                    "role": point.payload.get("role") if point.payload else None,
                    "similarity": point.score,
                })

        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[Timing] episodic_similarity_ms={elapsed_ms:.1f}")
        return memories

    def resolve_memory_for_storage(self, memory: MemoryBlob, limit: int = EPISODIC_TOP_K,
                                   threshold: float = EPISODIC_THRESHOLD):
        """Resolve the canonical memory to store without persisting anything yet."""
        policy = self._resolve_dedup_policy(memory)

        if policy == "document":
            exact_duplicate = self._find_exact_duplicate(memory)
            if exact_duplicate:
                return {
                    "action": "duplicate",
                    "memory_id": exact_duplicate.id,
                    "matched_id": exact_duplicate.id,
                    "similarity": 1.0,
                    "memory_blob": exact_duplicate,
                }

            return {
                "action": "store",
                "memory_id": memory.id,
                "matched_id": None,
                "similarity": 0.0,
                "memory_blob": memory,
            }

        if memory.embedding is None:
            memory.create_embedding()

        similar_memories = self.find_similar_memories(
            memory.content,
            query_embedding=memory.embedding,
            limit=limit,
            threshold=threshold,
        )

        if not similar_memories:
            return {
                "action": "store",
                "memory_id": memory.id,
                "matched_id": None,
                "similarity": 0.0,
                "memory_blob": memory,
            }

        best_match = similar_memories[0]
        if best_match.get("similarity", 0.0) < threshold:
            return {
                "action": "store",
                "memory_id": memory.id,
                "matched_id": None,
                "similarity": best_match.get("similarity", 0.0),
                "threshold": threshold,
                "memory_blob": memory,
            }

        if best_match.get("similarity", 0.0) < EPISODIC_MERGE_THRESHOLD:
            return {
                "action": "store",
                "memory_id": memory.id,
                "matched_id": best_match["id"],
                "similarity": best_match.get("similarity", 0.0),
                "threshold": threshold,
                "merge_threshold": EPISODIC_MERGE_THRESHOLD,
                "memory_blob": memory,
            }

        merged_content = self._merge_content_with_llm(best_match["content"], memory.content)
        merged_memory = MemoryBlob(
            content=merged_content,
            role=memory.role or best_match.get("role"),
            memory_type=memory.memory_type or best_match.get("memory_type"),
            id=best_match["id"],
            created_at=best_match["created_at"],
            tags=best_match.get("tags", {}),
        )
        merged_memory.frequency = best_match.get("frequency", 1) + 1
        merged_memory.salience = max(best_match.get("salience", 0.0), memory.salience)
        merged_memory.version = best_match.get("version", 1) + 1
        merged_memory.last_accessed = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "action": "update",
            "memory_id": merged_memory.id,
            "matched_id": best_match["id"],
            "similarity": best_match.get("similarity", 0.0),
            "threshold": threshold,
            "merge_threshold": EPISODIC_MERGE_THRESHOLD,
            "memory_blob": merged_memory,
        }

    def store_memory_with_dedup(self, memory: MemoryBlob, limit: int = EPISODIC_TOP_K,
                                threshold: float = EPISODIC_THRESHOLD):
        """Store a memory after checking for content-only duplicates.

        If the best match meets the similarity threshold, send the old and new
        content to the LLM, then update that older memory in place with the
        merged result. Otherwise store a brand-new record.
        """
        resolution = self.resolve_memory_for_storage(memory, limit=limit, threshold=threshold)
        self.persist_memory(resolution["memory_blob"])
        return resolution
    
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