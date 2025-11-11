import openai 
import dotenv
import os
dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.Client(api_key=openai_api_key)