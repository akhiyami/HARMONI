"""
This module contains the configuration for various vision models used in the application.
It includes models for face detection, landmark detection, audio transcription, emotion detection, and user retrieval
"""

#--------------------------------------- Imports ---------------------------------------#

import sys
import os

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

import dlib
import dotenv
import whisper

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    SiglipVisionModel,
    SiglipImageProcessor,
)
from insightface.app  import FaceAnalysis
from insightface.model_zoo import get_model as insightface_get_model

from config.utils import suppress_stdout

################
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

#--------------------------------------- Configuration ---------------------------------------#

# Load YOLOv8 face detector (deprecated, insightface used instead)
YOLO_FACE_MODEL = YOLO(
    hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
)

# Load Whisper model
WHISPER_MODEL = whisper.load_model("turbo")  # or parameterize this

# Load facial emotion model
EMOTION_PROCESSOR = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
EMOTION_MODEL = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")


# ULIP-p16 (deprecated, use insightface instead)
model_name = "hamedrahimi/ULIP-p16"
USER_RETRIEVER_MODEL = SiglipVisionModel.from_pretrained(model_name)
USER_RETRIEVER_PROCESSOR = SiglipImageProcessor.from_pretrained(model_name)

# INSIGHTFACE
with suppress_stdout():
    INSIGHTFACE_MODEL = insightface_get_model('buffalo_l', download=True)
    if INSIGHTFACE_MODEL is None:
        app = FaceAnalysis(name='buffalo_l')
        INSIGHTFACE_MODEL = insightface_get_model('buffalo_l', download=True)
    INSIGHTFACE_MODEL.prepare(ctx_id=0)  # or -1 for CPU