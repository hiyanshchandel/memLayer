from clients.vector_client import vec_client
import sqlite3
import json
from memory_blob.definition import MemoryBlob
from config import  SEMANTIC_COLLECTION_NAME, SEMANTIC_MEMORY_DB, SEMANTIC_THRESHOLD, SEMANTIC_TOP_K, neo4j_extraction_prompt
from clients.graphdb_client import graphdb_client
from clients.openai_client import openai_client
import re

class GraphMemoryManager:
    def __init__(self):
        self.client = graphdb_client
        self.openai_client = openai_client

    def extract_entities_and_relationships(self, role = neo4j_extraction_prompt, Memory = MemoryBlob) -> dict:
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": Memory.content}
            ]
        )
        ans = response.choices[0].message.content
        clean_json_str = re.sub(r"^```json\s*|\s*```$", "", ans.strip())
        data = json.loads(clean_json_str)
        return data
    
    def push_to_graphdb(self, Memory = MemoryBlob):
        data = self.extract_entities_and_relationships(role = neo4j_extraction_prompt, Memory = Memory)
        entity_lookup = {e["id"]: e for e in data["entities"]}

        with self.client.session() as session:
            # Create all nodes
            for entity in data["entities"]:
                session.run(
                    f"MERGE (n:{entity['label']} {{name: $name}})",
                    name=entity["name"]
                )

            # Create relationships
            for relation in data["relationships"]:
                start = entity_lookup[relation["start_id"]]
                end = entity_lookup[relation["end_id"]]
                rel_type = relation["type"]

                session.run(
                    f"""
                    MATCH (a:{start['label']} {{name: $start_name}}),
                        (b:{end['label']} {{name: $end_name}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    """,
                    start_name=start["name"],
                    end_name=end["name"]
                )
        return
        





    
