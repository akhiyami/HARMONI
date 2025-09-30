"""
# app.py
This file contains the main FastAPI application for handling user interactions, 
including setting users, asking questions, and retrieving answers from the LLM.
It integrates with the memory module to store and retrieve user-specific data.
It also handles user image for face recognition and user identification.
==========
Run the application in the root dir using:
    uvicorn conv_interface.app:app --reload
"""

#--------------------------------------- Imports ---------------------------------------#

import os
import time
import json
import yaml
import threading
import warnings
import sqlite3
from collections import deque
from io import BytesIO

from PIL import Image

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import get_face_embedding_model
from config.settings import LEN_HISTORY

from conversation.llm.openai_inferences import generate_answer, update_memory_llm
from conversation.memory.memory import user_retriever, update_memory, memory_retriever
from conversation.memory.utils import create_table, empty_database

from vision.detection import detect_faces_image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

#--------------------------------------- Configuration ---------------------------------------#

# Database connection
database = 'conv_interface/users.db'
conn = sqlite3.connect(database)
create_table(conn)      # Create the embeddings table if it does not exist

# Init FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="conv_interface/static"), name="static")

# Global variables for user session management
current_user = None
old_sessions = []
current_session = []

current_context = ""

# facial encoding model
model_config = get_face_embedding_model("INSIGHTFACE")  # or "ULIP-p16" for Siglip

config_path = "config/config.yaml"

#--------------------------------------- Routes ---------------------------------------#

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """
    Serve the main HTML page for the application.
    """
    with open("conv_interface/static/index.html", "r") as f:
        return f.read()
    
@app.get("/config")
def get_config():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config 


@app.post("/ask")
async def ask(question: str = Form(...)):
    """
    Handle the user's question, retrieve the answer from the LLM, and update the user's memory.
    """
    global current_session, current_user, conn, current_context

    all_start = time.time()

    # Check if a user is set
    if current_user is None:
        return {"error": "No user set. Please upload an image to set the user."}
    # Validate the question
    if not question.strip():
        return {"error": "Question cannot be empty."}
    

    # Update the user's memory in parallel with answer generation
    def update_user_memory():
        start_time = time.time()
        conn_thread = sqlite3.connect(database)

        try:
            new_memory_object = update_memory_llm(question, conn_thread, current_user, stm=current_session)
            nonlocal memory_user
            memory_user = update_memory(new_memory_object, current_user, conn_thread)
        finally:
            conn_thread.close()  # Ensure the connection is closed after use

        end_time = time.time()
        print(f"Memory updated in {end_time - start_time:.2f} seconds")


    memory_user = None
    memory_thread = threading.Thread(target=update_user_memory)
    memory_thread.start()

    # Generate the answer using the LLM
    context = current_context 
    answer, _ = generate_answer(question, current_session, context, conn, current_user)
    # Update the session history
    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})


    # Wait for the memory update thread to finish
    memory_thread.join()
    print(f"All ask_llm route takes {time.time() - all_start:.2f} seconds")

    return {"answer": answer, "profile": memory_user}
 

@app.post("/set_user")
async def set_user(image: UploadFile = File(...)):
    """
    Set the current user based on the uploaded image.
    This function performs face recognition to identify the user and retrieves their memory.
    """
    global current_session, current_user, conn

    # Receive the image from the application
    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    new_user, new_memory = user_retriever(img, conn, model_config)

    if new_user != current_user :# Check if a new user was identified

        # Update the global variables and reset the session
        old_sessions.append({"user": current_user, "session": current_session})

        current_user = new_user
        user_memory = new_memory
        current_session = []
        return {"user_id": current_user, "profile": user_memory}
    
    else:
        memory = memory_retriever(current_user, conn)
        return {"user_id": current_user, "profile": memory}
    
@app.post("/reset_session")
async def reset_session():
    global current_session, current_user
    old_sessions.append({"user": current_user, "session": current_session})
    current_session = []
    return {"status": "success"}

@app.post("/reset_database")
async def reset_database():
    conn = sqlite3.connect(database)

    empty_database(conn)

    conn = sqlite3.connect(database) 
    create_table(conn) 

    conn.close()
    return {"status": "success"}

@app.post("/restart_system")
async def restart_system():
    global current_session, current_user, old_sessions, current_context, conn

    current_user = None
    old_sessions = []
    current_session = []
    current_context = ""

    conn.close()
    conn = sqlite3.connect(database)

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


@app.post("/get_profile")
async def get_profile(
    user_id: str = Form(...),
    ):
    """
    Get the profile information for a specific user.
    """
    global conn
    profile = memory_retriever(user_id, conn)
    return {"profile": profile}




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
    
    
@app.post("/face_detection")
async def face_detection(image: UploadFile = File(...)):
    """
    Perform face detection on the uploaded image.
    """
    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Call the face detection function
    bounding_boxes = detect_faces_image(img)
    if len(bounding_boxes) == 0:
        return {"error": "No faces detected."}
    
    elif len(bounding_boxes) > 1:
        return {"error": "Multiple faces detected. Please upload an image with a single face."}

    bounding_box = bounding_boxes[0]
    bounding_box = {
        "x": int(bounding_box[0]),
        "y": int(bounding_box[1]),
        "width": int(bounding_box[2] - bounding_box[0]),
        "height": int(bounding_box[3] - bounding_box[1])
    }

    return {"bounding_box": bounding_box}



@app.post("/save_experiment")
async def save_experiment():
    """
    Save the current conversation experiment to a file.
    """
    global current_session, current_user, old_sessions, current_context

    sessions = old_sessions.copy()
    sessions.append({"user": current_user, "session": current_session})

    profile = memory_retriever(current_user, conn)

    context = current_context

    experience = {
        "user_id": current_user,
        "profile": profile,
        "context": context,
        "sessions": sessions
    }

    # Save the experience to a file
    result_path = "results/experiences"
    os.makedirs(result_path, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    with open(f"{result_path}/experience_{timestamp}.json", "w") as f:
        json.dump(experience, f, indent=4, ensure_ascii=False)

    return {"status": "success"}