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
from transformers import SiglipVisionModel, SiglipImageProcessor

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from llm.openai_utils import ask_llm
from memory.memory import user_retriever, update_memory, memory_retriever
from config import LEN_HISTORY
from memory.utils import create_table

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

# facial encoding model
model_name = "hamedrahimi/ULIP-p16"
model = SiglipVisionModel.from_pretrained(model_name)
processor = SiglipImageProcessor.from_pretrained(model_name)
model.eval()


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
    """
    global current_session, current_user, conn

    # Check if a user is set
    if current_user is None:
        return {"error": "No user set. Please upload an image to set the user."}
    # Validate the question
    if not question.strip():
        return {"error": "Question cannot be empty."}
    
    # Generate the answer using the LLM
    output = ask_llm(question, current_session, conn, current_user)
    answer = output.answer
    new_memory_object = output.updated_memory

    # Update the user's memory in the database
    memory_user = update_memory(new_memory_object, current_user, conn)

    # Update the session history
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
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    new_user, new_memory = user_retriever(img, conn, processor, model)

    if new_user != current_user :# Check if a new user was identified

        # Update the global variables and reset the session
        current_user = new_user
        user_memory = new_memory
        current_session = deque(maxlen=LEN_HISTORY)

        return {"user_id": current_user, "profile": user_memory}
    
    else:
        memory = memory_retriever(current_user, conn)
        return {"user_id": current_user, "profile": memory}