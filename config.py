import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


LEN_HISTORY = 10
DISTANCE_THRESHOLD = 0.4  # Distance threshold for user retrieval