"""
# app.py
This file contains the main FastAPI application for handling user interactions, 
including setting users, asking questions, and retrieving answers from the LLM.
It integrates with the memory module to store and retrieve user-specific data.
It also handles user image for face recognition and user identification.
==========
Routes:
- `/`: Serves the main HTML page.
- `/ask`: Handles user questions, retrieves answers from the LLM, and updates the user's memory.
- `/set_user`: Sets the current user based on the uploaded image, performing face recognition to identify the user and retrieve their memory.
==========
Run the application using:
    uvicorn app:app --reload
"""

#--------------------------------------- Imports ---------------------------------------#

import threading
import warnings
import sqlite3
from collections import deque
from io import BytesIO
import time

from PIL import Image

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import get_face_embedding_model
from config.settings import LEN_HISTORY

from conversation.llm.openai_inferences import generate_answer, update_memory_llm
from conversation.memory.memory import user_retriever, update_memory, memory_retriever
from conversation.memory.utils import create_table

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
current_session = deque(maxlen=LEN_HISTORY)

# facial encoding model
model_config = get_face_embedding_model("INSIGHTFACE")  # or "ULIP-p16" for Siglip

#--------------------------------------- Routes ---------------------------------------#

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """
    Serve the main HTML page for the application.
    """
    with open("conv_interface/static/index.html", "r") as f:
        return f.read()


@app.post("/ask")
async def ask(question: str = Form(...)):
    """
    Handle the user's question, retrieve the answer from the LLM, and update the user's memory.
    """
    global current_session, current_user, conn

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
    context = None
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
        current_user = new_user
        user_memory = new_memory
        current_session = deque(maxlen=LEN_HISTORY)

        return {"user_id": current_user, "profile": user_memory}
    
    else:
        memory = memory_retriever(current_user, conn)
        return {"user_id": current_user, "profile": memory}