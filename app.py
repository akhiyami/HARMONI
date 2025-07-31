"""
# app.py

"""

#--------------------------------------- Imports ---------------------------------------#

import threading
import warnings
import sqlite3
from collections import deque
from io import BytesIO
import time
import numpy as np
import os
from PIL import Image
import cv2
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from conversation.llm.openai_utils import generate_answer, update_memory_llm
from conversation.memory.memory import user_retriever, update_memory, memory_retriever
from conversation.memory.utils import create_table
from conversation.config.settings import LEN_HISTORY
from conversation.config import models as conversation_models

from vision.config import models as vision_models
from vision.detection import detect_speaking_face
from vision.audio import extract_and_transcribe_audio
from vision.emotions import detect_emotions

import utils 
import base64
import io
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

#--------------------------------------- Configuration ---------------------------------------#

# Init FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for user session management
# Load models from vision module
model = vision_models.YOLO_FACE_MODEL
landmark_detector = vision_models.LANDMARK_DETECTOR
whisper_model = vision_models.WHISPER_MODEL
emotion_model = vision_models.EMOTION_MODEL
emotion_processor = vision_models.EMOTION_PROCESSOR

# Load user retriever model and processor
user_retriever_model = conversation_models.USER_RETRIEVER_MODEL
user_retriever_processor = conversation_models.USER_RETRIEVER_PROCESSOR


# Database connection for user retrieval
database = 'users.db'
conn = sqlite3.connect(database)
create_table(conn)    

current_session = []

#--------------------------------------- Routes ---------------------------------------#

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """
    Serve the main HTML page for the application.
    """
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/set_video")
async def set_video(video: UploadFile = File(...)):
    """
    Handle the user's video input, extract audio, transcribe it, and update the user's memory.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    # Now you can open the video using OpenCV
    cap = cv2.VideoCapture(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_face = executor.submit(detect_speaking_face, cap, model, landmark_detector, save_frames=True)
        future_transcript = executor.submit(extract_and_transcribe_audio, tmp_path, whisper_model)

        speaking_face_row, grid, probs = future_face.result()
        transcript = future_transcript.result()

    user_image = speaking_face_row[0]

    with ThreadPoolExecutor(max_workers=2) as executor:

        future_emotion = executor.submit(detect_emotions, speaking_face_row, emotion_model, emotion_processor)
        future_user = executor.submit(user_retriever, user_image, None, user_retriever_processor, user_retriever_model, database)

        emotion, prob, emotions, probs = future_emotion.result()
        detected_user, memory_user = future_user.result()

    # Convert user_image (numpy array) to base64 string for sharing via JSON
    

    if isinstance(user_image, np.ndarray):
        img_pil = Image.fromarray(user_image)
    elif isinstance(user_image, Image.Image):
        img_pil = user_image.convert("RGB")  # Ensure it's in RGB format  

    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    user_image_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return {
        "user_image": user_image_base64,
        "detected_user": detected_user,
        "profile": memory_user,
        "emotion": emotion,
        "transcript": transcript
    }

@app.post("/answer")
async def answer_question(
            emotion: str = Form(...),
            current_user: str = Form(...),
            question: str = Form(...)
        ):
    """
    Process the user's question, update their memory, and return the response.
    """
    global current_session
    # Update memory with emotion and question
    def update_user_memory():
        start_time = time.time()
        conn_thread = sqlite3.connect(database)

        try:
            new_memory_object = update_memory_llm(question, current_session, conn_thread, current_user)
            global memory_user
            memory_user = update_memory(new_memory_object, current_user, conn_thread)
        finally:
            conn_thread.close()  # Ensure the connection is closed after use

        end_time = time.time()
        print(f"Memory updated in {end_time - start_time:.2f} seconds")

    memory_thread = threading.Thread(target=update_user_memory)
    memory_thread.start()

    visual_profile = {
        "emotion": emotion,
        "gender": None,
        "age": None,
    }

    context = ""

    # Generate the answer using the LLM
    answer, _ = generate_answer(question, current_session, context, conn, current_user, visual_profile)
    memory_thread.join()

    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})

    print(f"Answer generated: {answer}")

    return {
        "answer": answer,
        "profile": memory_user,
    }

    