from openai import OpenAI
import os
from dotenv import load_dotenv
from utils.retry import MaxRetriesExceeded, Retry

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

query = input("What would you like to ask your second brain?: ")

with client.responses.create(
    model="gpt-5",
    
    input=[
        {"role": "system", "content": "You are my second brain, and you know everytjing about AI engineering"},
        {
            "role": "user",
            "content": query,
        },
    ],
    stream=True,
) as stream:

    for event in stream:
        if event.type == "response.refusal.delta":
            print(event.delta, end="")
        elif event.type == "response.output_text.delta":
            print(event.delta, end="")
        elif event.type == "response.error":
            print(event.error, end="")
    