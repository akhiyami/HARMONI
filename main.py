import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import sqlite3
import time
import threading
from collections import deque

from vision_module.config import models as vision_models
from vision_module.vision import detect_speaking_face
from vision_module.audio import extract_and_transcribe_audio
from vision_module.emotions import detect_emotions

from conversation_module.memory.memory import user_retriever, update_memory
from conversation_module.llm.openai_utils import generate_answer, update_memory_llm
from conversation_module.memory.utils import create_table
from conversation_module.config import models as conversation_models
from conversation_module.config.settings import LEN_HISTORY

# Define the name of the video to process
name_video = 'vision_module/videos/sample4'  # Change this to your video name without extension

# Load models from vision module
model = vision_models.YOLO_FACE_MODEL
landmark_detector = vision_models.LANDMARK_DETECTOR
whisper_model = vision_models.WHISPER_MODEL
emotion_model = vision_models.EMOTION_MODEL
emotion_processor = vision_models.EMOTION_PROCESSOR

# Load user retriever model and processor
user_retriever_model = conversation_models.USER_RETRIEVER_MODEL
user_retriever_processor = conversation_models.USER_RETRIEVER_PROCESSOR

# Database connection for user retrieval
database = 'users.db'
conn = sqlite3.connect(database)
create_table(conn)    

if __name__ == "__main__":

    # Empty result folder if exists
    # Open video
    input_path = f'{name_video}.mp4'
    cap = cv2.VideoCapture(input_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_face = executor.submit(detect_speaking_face, cap, model, landmark_detector, save_frames=True)
        future_transcript = executor.submit(extract_and_transcribe_audio, f'{name_video}.mp4', whisper_model)

        speaking_face_row = future_face.result()
        transcript = future_transcript.result()

    user_image =  speaking_face_row[0]

    # Pick 10 frames equally spaced
    if len(speaking_face_row) > 10:
        speaking_face_row_array = np.array(speaking_face_row, dtype=object)
        sample_speaking_face_row = speaking_face_row_array[np.linspace(0, len(speaking_face_row_array) - 1, 10, dtype=int)].tolist()
    else:
        sample_speaking_face_row = speaking_face_row

    with ThreadPoolExecutor(max_workers=2) as executor:

        future_emotion = executor.submit(detect_emotions, sample_speaking_face_row, emotion_model, emotion_processor)
        future_user = executor.submit(user_retriever, user_image, None, user_retriever_processor, user_retriever_model, database)

        emotion, prob = future_emotion.result()
        detected_user, memory_user = future_user.result()


    question = transcript
    current_session = deque(maxlen=LEN_HISTORY)
    current_user = detected_user
    memory_user = None 

    def update_user_memory():
        start_time = time.time()
        conn_thread = sqlite3.connect(database)

        try:
            new_memory_object = update_memory_llm(question, current_session, conn_thread, current_user)
            global memory_user
            memory_user = update_memory(new_memory_object, current_user, conn_thread)
        finally:
            conn_thread.close()  # Ensure the connection is closed after use

        end_time = time.time()
        print(f"Memory updated in {end_time - start_time:.2f} seconds")

    memory_thread = threading.Thread(target=update_user_memory)
    memory_thread.start()

    # Generate the answer using the LLM
    answer = generate_answer(question, current_session, conn, current_user)
    memory_thread.join()

    print(f"Answer generated: {answer}")

