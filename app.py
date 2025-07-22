"""
# app.py
This file contains the main FastAPI application for handling user interactions, 
including setting users, asking questions, and retrieving answers from the LLM.
It integrates with the memory module to store and retrieve user-specific data.
It also handles user image for face recognition and user identification.
"""

##########
# Import necessary libraries
##########

import os
import json
import threading
import warnings
import sqlite3
from collections import deque
from io import BytesIO

import numpy as np
from PIL import Image

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import face_recognition

from llm.openai_utils import ask_llm
from memory.memory import user_retriever, update_memory
from config import LEN_HISTORY
from memory.utils import create_table
from models.frustration_predictor import get_frustration_predictor

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")


##########
# Configuration
##########

# Database connection
database = 'memory/users.db'
conn = sqlite3.connect(database)
create_table(conn)      # Create the embeddings table if it does not exist

# Init FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for user session management
current_user = None
current_session = deque(maxlen=LEN_HISTORY)


###########
## Routes
###########

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """
    Serve the main HTML page for the application.
    """
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/ask")
async def ask(question: str = Form(...)):
    """
    Handle the user's question, retrieve the answer from the LLM, and update the user's memory.
    Now includes parallel frustration prediction.
    """
    global current_session, current_user, conn

    # Check if a user is set
    if current_user is None:
        return {"error": "No user set. Please upload an image to set the user."}
    # Validate the question
    if not question.strip():
        return {"error": "Question cannot be empty."}
    
    # 🆕 PARALLEL FRUSTRATION PREDICTION
    # Prepare conversation history for prediction
    temp_history = list(current_session) + [{"role": "user", "content": question}]
    
    # Run frustration prediction in parallel with LLM
    frustration_predictor = get_frustration_predictor()
    will_be_frustrated = frustration_predictor.predict_will_be_frustrated(temp_history)
    
    # Generate the answer using the LLM (existing flow)
    output = ask_llm(question, current_session, conn, current_user)
    answer = output.answer
    new_memory_object = output.updated_memory

    # 🆕 ADD FRUSTRATION PREDICTION TO MEMORY
    from llm.openai_utils import Feature
    
    frustration_feature = Feature(
        name="will_be_frustrated",
        description="AI prediction of whether the user will be frustrated in the next interaction",
        tags=["prediction", "emotion", "frustration"],
        value=[str(will_be_frustrated).lower()]  # "true" or "false"
    )
    
    # Add to memory updates
    new_memory_object.append(frustration_feature)

    # Update the user's memory in the database (existing flow)
    memory_user = update_memory(new_memory_object, current_user, conn)

    # Update the session history (existing flow)
    current_session.append({"role": "user", "content": question})
    current_session.append({"role": "assistant", "content": answer})

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
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)

    # Perform face recognition to get the encodings
    encodings = face_recognition.face_encodings(img_array)
    if not encodings:
        return {"error": "No face detected in the image."}
    
    # Use the first encoding for user identification
    encodings = [encodings[0]]
    new_user, new_memory = user_retriever(encodings, conn)

    if new_user != current_user :# Check if a new user was identified

        # Update the global variables and reset the session
        current_user = new_user
        user_memory = new_memory
        current_session = deque(maxlen=LEN_HISTORY)

        return {"user_id": current_user, "profile": user_memory}