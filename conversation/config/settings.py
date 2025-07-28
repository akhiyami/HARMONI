"""
This module contains configuration settings for the conversation module.
"""

#--------------------------------------- Imports ---------------------------------------#

import os
from dotenv import load_dotenv


#--------------------------------------- Configuration Settings ---------------------------------------#

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Default configuration values
LEN_HISTORY = 10
RECOGNITION_THRESHOLD = 0.85  # Distance threshold for user retrieval
NUM_TOP_FEATURES = 5 # Number of top features to retrieve for context