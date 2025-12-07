from mistralai import Mistral

import dotenv
import os

dotenv.load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

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

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    #return response.choices[0].message["content"]
    return response.choices[0].message

    #return "This is a placeholder answer."
