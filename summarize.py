from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

def summarize_text(texts: list[str]) -> dict:  # Changed return type to dict
    if not texts:
        return {"insights": []}  # Return dict directly
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Join list of texts into a single string
    combined_text = "\n".join(texts) if isinstance(texts, list) else texts
    
    system_prompt = """You are a memory extraction system. Your job is to analyze conversation messages and extract only the most valuable, memorable insights.

Rules:
1. Extract ONLY information worth remembering long-term (facts, preferences, important context, key decisions)
2. Ignore small talk, greetings, acknowledgments, and transactional messages
3. Separate insights by who they're about: AI behavior/responses vs User preferences/information
4. Group related information into concise points
5. If multiple topics are discussed, create separate points for each
6. If nothing is worth remembering, return empty lists
7. Each point should be clear, specific, and self-contained

Output Format (JSON):
{
  "ai_insights": [
    "AI provided code examples for Python functions",
    "AI suggested using embeddings for semantic search"
  ],
  "user_insights": [
    "User prefers concise explanations over verbose ones",
    "User is building a memory layer for chatbots"
  ]
}

If nothing is memorable, return: {"ai_insights": [], "user_insights": []}"""
    
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_text}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)  # Parse JSON string to dict