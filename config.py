embedding_model = "text-embedding-3-large"  
semantic_extraction_model = "gpt-4o-mini"

EPISODIC_MEMORY_DB = "episodic_memory.db"
EPISODIC_COLLECTION_NAME = "episodic_memories"
EPISODIC_THRESHOLD = 0.82
EPISODIC_MERGE_THRESHOLD = 0.95
EPISODIC_TOP_K = 5
EPISODIC_RETRIEVAL_FALLBACK_THRESHOLDS = (0.82, 0.65, 0.5, 0.35, None)

DOCUMENT_MEMORY_TYPES = {
  "document_chunk",
  "bug_story",
  "article",
  "pdf_text",
  "note",
  "manual_text",
}

CHAT_MEMORY_TYPES = {
  "chat_message",
  "conversation_memory",
  "user_fact",
  "assistant_message",
}

SEMANTIC_MEMORY_DB = "semantic_memory.db"
SEMANTIC_COLLECTION_NAME = "semantic_memories"
SEMANTIC_THRESHOLD = 0.75
SEMANTIC_TOP_K = 5

GRAPH_PUSH_MIN_INTERVAL_SECONDS = 1.0
GRAPH_PUSH_MAX_RETRIES = 5
GRAPH_PUSH_RETRY_BASE_SECONDS = 0.75
GRAPH_CONTEXT_ENTITY_LIMIT = 20
GRAPH_CONTEXT_CACHE_LIMIT = 100

NEO4J_URI = "neo4j://localhost:7687"
neo4j_extraction_prompt = """Extract a compact, canonical knowledge graph from the input.

Return ONLY valid JSON in this exact shape:
{
  "entities": [
    {"id": "1", "label": "Person|Project|Technology|Algorithm|Organization|Location|Concept|Event|Document|Attribute", "name": "canonical name"}
  ],
  "relationships": [
    {"start_id": "1", "end_id": "2", "type": "works_on|studies_for|created|implemented|uses|belongs_to|part_of|located_in|goes_to|has_to_go|worried_about|related_to|alias_of|same_as|has_attribute|described_as|causes|caused_by|leads_to|results_in|explains|requires"}
  ]
}

Rules:
- Use only explicit facts from the input.
- Never invent relationships or story-like verbs.
- Prefer canonical names and merge aliases when obvious.
- If unsure about the relation type, use related_to.
- If unsure about the entity label, use Document for text artifacts and Concept for abstract ideas.
- Keep the graph conservative and compact.
- Return empty arrays if nothing explicit is present.
"""
