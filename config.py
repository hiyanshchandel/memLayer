embedding_model = "text-embedding-3-large"  

EPISODIC_MEMORY_DB = "episodic_memory.db"
EPISODIC_COLLECTION_NAME = "episodic_memories"
EPISODIC_THRESHOLD = 0.20
EPISODIC_TOP_K = 5

SEMANTIC_MEMORY_DB = "semantic_memory.db"
SEMANTIC_COLLECTION_NAME = "semantic_memories"
SEMANTIC_THRESHOLD = 0.75
SEMANTIC_TOP_K = 5

NEO4J_URI = "neo4j://localhost:7687"
neo4j_extraction_prompt = role = """You are the knowledge base creator for a graph database. 
Using the given conversation, extract entities (such as people, organizations, events, concepts, tools, datasets, etc.) and their relationships (such as created, works_on, uses, belongs_to, is_part_of, studies, etc.) that can be stored in a graph database.

You must only reply in JSON format as shown in the example below:

{
  "entities": [
    {"id": "1", "label": "Person", "name": "Hiyansh"},
    {"id": "2", "label": "Project", "name": "CycleGAN Horse-Zebra"},
    {"id": "3", "label": "Algorithm", "name": "Prim's MST"}
  ],
  "relationships": [
    {"start_id": "1", "end_id": "2", "type": "created"},
    {"start_id": "1", "end_id": "3", "type": "implemented"}
  ]
}
The JSON must contain two main keys:
- "entities" → a list of all unique entities detected from the conversation
- "relationships" → a list of relationships connecting those entities

Keep names concise, ensure no duplicates, and use lowercase for relationship types.
Do not include any text or explanation outside the JSON."""
