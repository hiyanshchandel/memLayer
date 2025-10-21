from openai import OpenAI
from config import embedding_model

def get_embedding(input_text):
    client = OpenAI()
    response = client.embeddings.create(
        input=input_text, 
        model=embedding_model, 
        dimensions=3072 
    )
    return response.data[0].embedding  