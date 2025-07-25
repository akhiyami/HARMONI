import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import whisper
import dlib
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load YOLOv8 face detector
YOLO_FACE_MODEL = YOLO(
    hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
)

# Load dlib landmark detector
LANDMARK_DETECTOR = dlib.shape_predictor(
    os.path.join('vision_module/models', 'shape_predictor_68_face_landmarks.dat') #TODO: fix path work in all environments
)

# Load Whisper model
WHISPER_MODEL = whisper.load_model("base")  # or parameterize this

# Load facial emotion model
EMOTION_PROCESSOR = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
EMOTION_MODEL = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")