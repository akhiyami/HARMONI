"""
This module contains the configuration settings for the vision module. 
"""

#--------------------------------------- Imports ---------------------------------------#

import os
from dotenv import load_dotenv
import torch

#--------------------------------------- Configuration ---------------------------------------#

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable.")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Please set the OPENAI_API_KEY environment variable if you want to use OpenAI models.")

# Device setup
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

##########
# Video settings
##########

# Frame buffer size
LEN_FRAME_BUFFER = 15
FRAME_STRIDE = 3 # Process every n frame to reduce computation
FILTER_THRESHOLD = 5  # Minimum number of frames to consider a face valid

##########
# Conversation settings
##########

LEN_HISTORY = 10
RECOGNITION_THRESHOLD = 0.85  # Distance threshold for user retrieval
NUM_TOP_FEATURES = 10 # Number of top features to retrieve for context