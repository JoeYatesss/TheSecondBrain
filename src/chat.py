import asyncio
import os
from dotenv import load_dotenv
from utils.retry import MaxRetriesExceeded, Retry
from utils.streaming import Client, Chat

load_dotenv()

async def main():

    query = input("What would you like to ask your second brain?: ")
    client = Client()
    chat = Chat(client)   # Holds the OpenAI connection          
    response = chat.get_response(query)    # Sends a message, returns response
    return response

if __name__ == '__main__':
    main()