import os
import json
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from llm.openai_utils import ask_llm
from memory.memory import user_retriever, load_users, save_user
from config import HISTORY_FILE, LEN_HISTORY

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import face_recognition
import threading
from collections import deque

# Init FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global memory and user
current_user = None
current_session = deque(maxlen=LEN_HISTORY)
users_data = load_users()

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/ask")
async def ask(question: str = Form(...)):
    global current_session, users_data
    user_id = current_user

    memory_user = users_data.get(user_id, {}).get("user_memory", [])

    output = ask_llm(question, current_session, memory_user)

    answer = output.answer

    new_memory_object = output.updated_memory
    new_memory = [{"name": item.name, "description": item.description} for item in new_memory_object]

    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})
    
    users_data[user_id]["user_memory"] = new_memory

    return {"answer": answer, "profile": new_memory}

@app.post("/set_user")
async def set_user(image: UploadFile = File(...)):
    global current_user, users_data, current_session
    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)
    encodings = face_recognition.face_encodings(img_array)

    new_user, new_memory = user_retriever(encodings, users_data)
    if new_user != current_user :

        save_user(users_data)

        current_user = new_user
        user_memory = new_memory


        current_session = []

        return {"user_id": current_user, "profile": user_memory}