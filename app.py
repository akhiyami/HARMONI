import os
import json
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from llm.openai_utils import ask_llm
from memory.memory import user_retriever, load_users, save_user, retrieve_memory
from config import LEN_HISTORY

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import face_recognition
import threading
from collections import deque

import sqlite3


database = 'memory/users.db'
conn = sqlite3.connect(database)


# Init FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global memory and user
current_user = None
current_session = deque(maxlen=LEN_HISTORY)

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/ask")
async def ask(question: str = Form(...)):
    global current_session
    user_id = current_user

    conn = sqlite3.connect(database)

    output = ask_llm(question, current_session, conn=conn, current_user=user_id)

    answer = output.answer
    new_memory_object = output.updated_memory

    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM {user_id}")
    existing_features = cursor.fetchall()

    for item in new_memory_object:
        if item.name not in [feature[0] for feature in existing_features]:
            cursor.execute(
                f"INSERT INTO {user_id} (name, description, tags, value) VALUES (?, ?, ?, ?)",
                (item.name, item.description, (";").join(item.tags), (";").join(item.value))
            )
            conn.commit()
        else:
            cursor.execute(
                f"UPDATE {user_id} SET description = ?, tags = ?, value = ? WHERE name = ?",
                (item.description, (";").join(item.tags), (";").join(item.value), item.name)
            )
            conn.commit()

    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})

    memory_user = retrieve_memory(user_id, conn)

    return {"answer": answer, "profile": memory_user}


@app.post("/set_user")
async def set_user(image: UploadFile = File(...)):
    global current_user, current_session
    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)
    encodings = face_recognition.face_encodings(img_array)

    new_user, new_memory = user_retriever(encodings, conn)

    if new_user != current_user :

        current_user = new_user
        user_memory = new_memory

        current_session = []

        return {"user_id": current_user, "profile": user_memory}