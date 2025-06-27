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
from config import HISTORY_FILE, LEN_HISTORY

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

    output = ask_llm(question, current_session, memory_user)

    print(output)

    answer = output.answer

    new_memory = output.updated_memory
    print(f"New memory for user {user_id}: {new_memory}")

    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})

    def update_and_reload():
        global current_session
        lt_interactions = current_session[0:-LEN_HISTORY]
        if not isinstance(lt_interactions, list):
            lt_interactions = [lt_interactions]
        update_ltm(user_id, lt_interactions)
        current_session = current_session[-LEN_HISTORY:]
        global users_data
        users_data = load_users()
        print(f"Updated long-term memory for user {user_id}")

    if len(current_session) > LEN_HISTORY:
        thread = threading.Thread(target=update_and_reload)
        thread.start()

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
                user = current_user
                update_ltm(current_user, current_session)
                global users_data
                users_data = load_users()
                print(f"Updated long-term memory for user {user}")

        thread = threading.Thread(target=update_and_reload)
        thread.start()

        current_user = new_user
        user_memory = new_memory

        current_session = []

        return {"user_id": current_user, "history": user_memory}