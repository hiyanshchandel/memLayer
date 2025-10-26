from embeddings import get_embedding
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from agents import Agent, Runner
load_dotenv()
import json
from .upsert_to_qdrant import upsert_vector

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = QdrantClient(
    host = "localhost",
    port = 6333,
)

from typing import Tuple

from typing import Tuple, List

def get_similar(text: str, threshold: float = 0.70) -> Tuple[List[str], List[str], List[float]]:
    """Get similar texts, their IDs, and similarity scores from vector DB."""
    query_embedding = get_embedding(text)
    result_list = []
    ids = []
    scores = []
    
    results = client.query_points(
        collection_name='memLayer',
        query=query_embedding,
        using='dense-vector',
        limit=3,
        score_threshold=threshold
    )

    for point in results.points:
        result_list.append(point.payload['text'])
        ids.append(point.id)
        scores.append(point.score)
    
    return result_list, ids, scores


def check_max_similarity(scores: List[float]) -> float:
    """Return the maximum similarity score from the list."""
    return max(scores) if scores else 0.0
 
 
mem_agent = Agent(
    name="Memory Decision Agent", 
    instructions="""You are a memory deduplication assistant. 
    
    Your task: Decide if new information should be stored as a NEW fact or MERGED with existing ones.

    Respond in JSON format:
    {
        "action": "store" or "merge",
        "reason": "brief explanation",
        "merges": [
            {
                "index": 0,
                "merged_text": "new text merged with fact at index 0"
            },
            {
                "index": 1,
                "merged_text": "new text merged with fact at index 1"
            }
        ] (only if action is merge - provide separate merged version for EACH fact)
    }

    If merging, create a separate merged_text for EACH existing fact you want to merge with.
    """
)

async def deduplication_decision(new_text: str, similar_facts: list):
    """JSON format:
    {
    "action": "store" or "merge",
    "reason": "brief explanation",
    "merges": [
        {
            "index": 0,
            "merged_text": "new text merged with fact at index 0"
        },
        {
            "index": 1,
            "merged_text": "new text merged with fact at index 1"
        }
    ] (only if action is merge - provide separate merged version for EACH fact)}"""
    # Format the similar facts clearly
    formatted_facts = "\n".join([f"{i}. {fact}" for i, fact in enumerate(similar_facts)])
    prompt = f"""New user information: {new_text}
    Existing similar facts in database:
    {formatted_facts}
    Should this be stored as a new fact or merged with existing ones? If merging with multiple facts, provide a separate merged version for each."""
    results = await Runner.run(mem_agent, prompt)
    results_json = json.loads(results.final_output)
    print(f"Deduplication decision: {results_json}")
    return results_json 

def merge_memory(json_data: dict, role: str, new_text : str, org_ids: list) -> str:
    """merge memory based on JSON decision data."""
    if json_data['action'] == 'store':
        upsert_vector(
            collection_name='memLayer',
            role=role,
            text=new_text
        )
        print("Storing as new fact.")
    if json_data['action'] == 'merge':
        for merge in json_data['merges']:
            merged_text = merge['merged_text']
            id  = org_ids[merge['index']]
            upsert_vector(
                collection_name='memLayer',
                role=role,
                text=merged_text,
                point_id=id
            )
            print(f"Merging with fact ID {id}.")

        

