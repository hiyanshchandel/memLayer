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
    
    system_prompt = """You are a memory extraction system. Your task is to analyze conversation messages and extract only the most valuable and memorable insights.
Guidelines:
1. Extract ONLY information worth remembering long-term (such as facts, preferences, important context, or key decisions).
2. Disregard small talk, greetings, acknowledgments, and transactional messages.
3. Clearly separate insights based on whom they pertain to: AI behavior/responses or User preferences/information.
4. Consolidate related information into concise, well-structured points.
5. If multiple topics are discussed, create distinct points for each.
6. If there are no noteworthy insights, return empty lists.
7. Each point must be clear, specific, and self-contained to ensure usefulness later.
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
If no memorable information is present, return: {"ai_insights": [], "user_insights": []}"""
    
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_text}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    print(f"Response from OpenAI: {response.choices[0].message.content}")
    return json.loads(response.choices[0].message.content)  # Parse JSON string to dict