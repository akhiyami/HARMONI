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

    new_memory_dict = [{"name": item.name, "description": item.description, "tags": item.tags, "value": item.value} for item in new_memory_object]

    for item in new_memory_dict:
        if item["name"] not in [feature["name"] for feature in memory_user]:
            memory_user.append(item)
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"INSERT INTO {user_id} (name, description, tags, value, embeddings) VALUES (?, ?, ?, ?, ?)",
                    (item["name"], item["description"], (";").join(item["tags"]), (";").join(item["value"]), None)
                )
                conn.commit()

        else:
            for feature in memory_user:
                if feature["name"] == item["name"]:
                    feature["description"] = item["description"]
                    feature["tags"] = item["tags"]
                    feature["value"] = item["value"]
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            f"UPDATE {user_id} SET description = ?, tags = ?, value = ? WHERE name = ?",
                            (item["description"], (";").join(item["tags"]), (";").join(item["value"]), item["name"])
                        )
                        conn.commit()
                    break



    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})

    users_data[user_id]["user_memory"] = memory_user

    return {"answer": answer, "profile": memory_user}

@app.post("/set_user")
async def set_user(image: UploadFile = File(...)):
    global current_user, users_data, current_session
    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)
    encodings = face_recognition.face_encodings(img_array)

    new_user, new_memory = user_retriever(encodings, users_data, conn)

    if new_user != current_user :

        save_user(users_data)

        current_user = new_user
        user_memory = new_memory


        current_session = []

        return {"user_id": current_user, "profile": user_memory}