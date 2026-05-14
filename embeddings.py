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


def get_embeddings(input_texts):
    client = OpenAI()
    response = client.embeddings.create(
        input=input_texts,
        model=embedding_model,
        dimensions=3072
    )
    return [item.embedding for item in response.data]