import os
import json
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from llm.openai_utils import ask_llm
from memory.memory import user_retriever, load_users, dump_interaction_history
from config import HISTORY_FILE

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import face_recognition

# Init FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global memory and user
current_user = None
users_data = load_users()
history_data = {}

if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        try:
            history_data = json.load(f)
        except json.JSONDecodeError:
            print("Erreur de décodage JSON dans le fichier d'historique.")

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/ask")
async def ask(question: str = Form(...)):
    global history_data
    user_id = current_user
    answer = ask_llm(question, history_data, user_id)

    history_data[user_id].append({"role": "user", "content": question})
    history_data[user_id].append({"role": "assistant", "content": answer})

    dump_interaction_history(history_data, HISTORY_FILE)

    return {"answer": answer, "history": history_data[user_id]}

@app.post("/set_user")
async def set_user(image: UploadFile = File(...)):
    global current_user, users_data
    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)
    encodings = face_recognition.face_encodings(img_array)

    current_user = user_retriever(encodings, users_data)
    user_history = history_data.get(current_user, [])
    return {"user_id": current_user, "history": user_history}
