"""
This module contains the configuration settings for the vision module. 
"""

#--------------------------------------- Imports ---------------------------------------#

import os
from dotenv import load_dotenv

#--------------------------------------- Configuration ---------------------------------------#

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

# Device setup
import torch
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Frame buffer size
LEN_FRAME_BUFFER = 15
FRAME_STRIDE = 5  # Process every 5th frame to reduce computation
FILTER_THRESHOLD = 5  # Minimum number of frames to consider a face valid
