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
import yaml

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import models, get_face_embedding_model
from config.settings import LEN_HISTORY

from conversation.llm.openai_inferences import generate_answer, update_memory_llm
from conversation.memory.memory import user_retriever, update_memory, memory_retriever
from conversation.memory.utils import create_table, empty_database


from vision.detection import detect_speaking_face
from vision.audio import extract_and_transcribe_audio
from vision.emotions import detect_emotions

import utils 
import base64
import io
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

#--------------------------------------- Configuration ---------------------------------------#

# Init FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for user session management
# Load models from vision module
model = models.YOLO_FACE_MODEL
landmark_detector = models.LANDMARK_DETECTOR
stt_model = models.WHISPER_MODEL
emotion_model = models.EMOTION_MODEL
emotion_processor = models.EMOTION_PROCESSOR
insightface_model = models.INSIGHTFACE_MODEL

# Load user retriever model and processor
user_retriever_config = get_face_embedding_model("INSIGHTFACE")

# Database connection for user retrieval
database = 'users.db'
conn = sqlite3.connect(database)
create_table(conn)   
conn.close() 

current_session = []
current_user_image = None
html_blocks = []
current_context = ""

config_path = "config/config.yaml"


#--------------------------------------- Routes ---------------------------------------#

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """
    Serve the main HTML page for the application.
    """
    with open("static/index.html", "r") as f:
        return f.read()
    

@app.get("/config")
def get_config():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config 


@app.post("/set_video")
async def set_video(video: UploadFile = File(...)):
    """
    Handle the user's video input, extract audio, transcribe it, and update the user's memory.
    """
    global html_blocks, current_user_image
    html_blocks = []  

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name


    with ThreadPoolExecutor(max_workers=2) as executor:
        future_face = executor.submit(detect_speaking_face, tmp_path, save_frames=True)
        future_transcript = executor.submit(extract_and_transcribe_audio, tmp_path, stt_model)

        speaking_face_row, grid, probs = future_face.result()
        transcript = future_transcript.result()

    html_blocks.append(utils.display_image_grid_html(grid, probs, np.argmax(probs), jupyter=False))

    user_image = speaking_face_row[0]
    
    with ThreadPoolExecutor(max_workers=2) as executor:

        future_emotion = executor.submit(detect_emotions, speaking_face_row, emotion_model, emotion_processor)
        future_user = executor.submit(user_retriever, user_image, None, user_retriever_config, database)

        emotion, prob, emotions, probs = future_emotion.result()
        detected_user, memory_user = future_user.result()

    # Convert user_image (numpy array) to base64 string for sharing via JSON
    if isinstance(user_image, np.ndarray):
        img_pil = Image.fromarray(user_image)
    elif isinstance(user_image, Image.Image):
        img_pil = user_image.convert("RGB")  # Ensure it's in RGB format  

    user_image = np.array(user_image)
    current_user_image = user_image

    html_blocks.append(utils.user_memory_to_html(memory_user, user_image, f"Detected User: {detected_user}", jupyter=False))
    html_blocks.append(utils.display_pie_chart(emotions, probs, emotion, prob, jupyter=False))
    html_blocks.append(utils.display_sequence_with_transcription(speaking_face_row, transcript, jupyter=False))

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
    global current_session, html_blocks, current_user_image
    # Update memory with emotion and question
    def update_user_memory():
        start_time = time.time()
        conn_thread = sqlite3.connect(database)

        try:
            new_memory_object = update_memory_llm(question, conn_thread, current_user, stm=current_session)
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

    context = current_context

    conn = sqlite3.connect(database)
    # Generate the answer using the LLM
    start_time = time.time()
    answer, retrieved_features = generate_answer(question, current_session, context, conn, current_user, visual_profile)
    end_time = time.time()
    memory_thread.join()


    conn.close()

    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})

    html_blocks.append(utils.display_answer(answer, memory_user, retrieved_features, end_time - start_time, jupyter=False))
    html_blocks.append(utils.user_memory_to_html(memory_user, current_user_image, f"Memory updated for {current_user}", title="Updated Memory", jupyter=False))

    html_text = ('\n').join([str(block) for block in html_blocks])

    return {
        "answer": answer,
        "profile": memory_user,
        "logs": html_text,
    }

@app.post("/reset_session")
async def reset_session():
    global current_session, html_blocks, current_user_image
    current_session = []
    html_blocks = []
    current_user_image = None
    return {"status": "success"}


@app.post("/reset_database")
async def reset_database():
    conn = sqlite3.connect(database)

    empty_database(conn)

    conn = sqlite3.connect(database) 
    create_table(conn) 

    conn.close()
    return {"status": "success"}

@app.get("/get_context")
async def get_context():
    """
    Get the current context for the user.
    """
    global current_context
    return {"context": current_context}

@app.post("/set_context")
async def set_context(context: str = Form(...)):
    """
    Update the context for the user.
    """
    global current_context
    current_context = context
    return {"status": "success"}

@app.post("/identify_user")
async def identify_user(user_image: UploadFile = File(...)):
    """
    Edit user information.
    """
    global current_user_image
    image_bytes = await user_image.read()

    current_user_image = Image.open(io.BytesIO(image_bytes))
    current_user_image = current_user_image.convert("RGB")
    current_user_image = np.array(current_user_image)

    detected_user, memory_user = user_retriever(current_user_image, None, user_retriever_config, database)
    
    return {
        "user": detected_user,
        "profile": memory_user
    }

@app.post("/edit_user")
async def edit_user(
    user_id: str = Form(...),
    profile_data: str = Form(...),
    ):
    """
    Edit user information in the database.
    """
    # Parse the JSON string sent from the frontend
    try:
        profile_dict = json.loads(profile_data)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON data: {str(e)}"}

    # update the user profile in the database here
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    # reset the user table
    cursor.execute(f"DELETE FROM {user_id}")
    conn.commit()

    # Insert the new profile data
    for feature in profile_dict:
        values_list = feature.get("value", [])
        value = ';'.join(values_list) if isinstance(values_list, list) else values_list
        cursor.execute(
            f"INSERT INTO {user_id} (type, name, description, value) VALUES (?, ?, ?, ?)",
            (feature.get("type"), feature.get("name"), feature.get("description"), value),
        )

    conn.commit()
    conn.close()
    return {"status": "success"}
    
    
