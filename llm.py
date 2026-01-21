#from mistralai import Mistral
#from geminiai import Gemini
from google import genai

import dotenv
import os

dotenv.load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()
#client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def answer_question(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    You are a helpful assistant. Use ONLY the information below to answer.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    If the answer is not in the context, say you don't know.
    """

    # Use generate_content for Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash", # or "gemini-1.5-pro"
        contents=prompt
    )

    return response.text

    #return response.choices[0].message["content"]
    return response.choices[0].message

    #return "This is a placeholder answer."
