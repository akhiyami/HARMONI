import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

# Device setup
import torch
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Frame buffer size
LEN_FRAME_BUFFER = 15
