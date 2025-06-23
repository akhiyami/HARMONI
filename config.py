import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

HISTORY_FILE = "memory/history.json"
USERS_FILE = "memory/users.json"
LEN_HISTORY = 10
