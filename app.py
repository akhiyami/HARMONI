import os
import json
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from llm.openai_utils import ask_llm
from memory.memory import user_retriever, load_users, update_ltm
from config import HISTORY_FILE

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import face_recognition
import threading

# Init FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global memory and user
current_user = None
current_session = []
users_data = load_users()


@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/ask")
async def ask(question: str = Form(...)):
    global current_session
    user_id = current_user

    memory_user = users_data.get(user_id, {}).get("user_memory", [])

    answer = ask_llm(question, current_session, memory_user)

    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})

    return {"answer": answer, "history": current_session}

@app.post("/set_user")
async def set_user(image: UploadFile = File(...)):
    global current_user, users_data, current_session
    users_data = load_users()
    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)
    encodings = face_recognition.face_encodings(img_array)

    new_user, new_memory = user_retriever(encodings, users_data)
    if new_user != current_user :

        def update_and_reload():
            if current_user is not None:
                update_ltm(current_user, current_session)
                # Signal: reload users_data after update
                global users_data
                users_data = load_users()

        thread = threading.Thread(target=update_and_reload)
        thread.start()

        current_user = new_user
        user_memory = new_memory

        current_session = []

        return {"user_id": current_user, "history": user_memory}