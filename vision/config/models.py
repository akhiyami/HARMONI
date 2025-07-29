"""
This module contains the configuration for various vision models used in the application.
It includes models for face detection, landmark detection, audio transcription, emotion detection, and user retrieval
"""

#--------------------------------------- Imports ---------------------------------------#

import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import whisper
import dlib
from transformers import AutoImageProcessor, AutoModelForImageClassification

#--------------------------------------- Configuration ---------------------------------------#

# Load YOLOv8 face detector
YOLO_FACE_MODEL = YOLO(
    hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
)

# Load dlib landmark detector
LANDMARK_DETECTOR = dlib.shape_predictor(
    os.path.join('vision/models', 'shape_predictor_68_face_landmarks.dat') 
)

# Load Whisper model
WHISPER_MODEL = whisper.load_model("base")  # or parameterize this

# Load facial emotion model
EMOTION_PROCESSOR = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
EMOTION_MODEL = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")