from openai import OpenAI
import os
from dotenv import load_dotenv
from utils.retry import MaxRetriesExceeded, Retry

load_dotenv()

class Client:
    def __init__(self):
        self.get_api_key()
        self.openai_client = OpenAI(api_key=self.api_key)
        

    def get_api_key(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return self.api_key
    
    def _get_client(self):
        return self.openai_client

class Chat:
    def __init__(self, client) -> None:
        self.client = client

    def get_response(self, query):
        retry = Retry(max_retries = 3)

        def make_api_call():
            return self.client._get_client().responses.create(
                model="gpt-5",
                
                input=[
                    {"role": "system", "content": "You are my second brain, and you know everything about AI engineering"},
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                stream=True,
            )

        with retry.call_with_retry(make_api_call) as stream:
            full_response = ''
            for event in stream:
                if event.type == "response.refusal.delta":
                    print(event.delta, end="")
                    full_response += event.delta
                elif event.type == "response.output_text.delta":
                    print(event.delta, end="")
                    full_response += event.delta
                elif event.type == "response.error":
                    print(event.error, end="")
                    full_response += event.error
        return full_response
