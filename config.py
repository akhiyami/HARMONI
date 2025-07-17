import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Default configuration values
LEN_HISTORY = 10
RECOGNITION_THRESHOLD = 0.9  # Distance threshold for user retrieval
NUM_TOP_FEATURES = 5 # Number of top features to retrieve for context