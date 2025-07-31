import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
import os
import webbrowser

from vision.config import models as vision_models
from vision.detection import detect_speaking_face
from vision.audio import extract_and_transcribe_audio
from vision.emotions import detect_emotions

from conversation.memory.memory import user_retriever, update_memory
from conversation.llm.openai_utils import generate_answer, update_memory_llm
from conversation.memory.utils import create_table
from conversation.config import models as conversation_models
from conversation.config.settings import LEN_HISTORY

import utils 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Choose if we save a html file with the results
SAVE_HTML = True
if SAVE_HTML:
    html_blocks = []

# Define the name of the video to process
name_video = 'data/20250728_123821'  # Change this to your video name without extension

if SAVE_HTML:
    html_blocks.append(utils.display_video(f"{name_video}.mp4", jupyter=False))

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

        speaking_face_row, grid, probs = future_face.result()
        transcript = future_transcript.result()

    if SAVE_HTML:
        html_blocks.append(utils.display_image_grid_html(grid, probs, np.argmax(probs), jupyter=False))

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

        emotion, prob, emotions, probs = future_emotion.result()
        detected_user, memory_user = future_user.result()

    if SAVE_HTML:
        user_image = np.array(user_image)
        html_blocks.append(utils.user_memory_to_html(memory_user, user_image, f"Detected User: {detected_user}", jupyter=False))
        html_blocks.append(utils.display_pie_chart(emotions, probs, emotion, prob, jupyter=False))
        html_blocks.append(utils.display_sequence_with_transcription(speaking_face_row, transcript, jupyter=False))
    
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

    visual_profile = {
        "emotion": emotion,
        "gender": None,
        "age": None,
    }

    context = ""

    # Generate the answer using the LLM
    start_time = time.time()
    answer, retrieved_features = generate_answer(question, current_session, context, conn, current_user, visual_profile)
    end_time = time.time()
    memory_thread.join()

    if SAVE_HTML:
        html_blocks.append(utils.display_answer(answer, memory_user, retrieved_features, end_time - start_time, jupyter=False))
        html_blocks.append(utils.user_memory_to_html(memory_user, user_image, f"Memory updated for {current_user}", title="Updated Memory", jupyter=False))

        utils.save_html_page(html_blocks, filename=f"output.html")
        print(f"HTML saved to output.html")

        # Open the HTML file in the default web browser
        webbrowser.open("output.html")

    if not SAVE_HTML:
        print(f"Answer generated: {answer}")

        # Printing and plotting
        n_frames = len(speaking_face_row)
        fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2, 2), constrained_layout=True)

        # Plot each frame
        for j in range(n_frames):
            ax = axes[j]
            ax.imshow(speaking_face_row[j])
            ax.axis('off')

        plt.suptitle(f'"{transcript}"', fontsize=16)
        plt.figtext(0.5, 0.01, f"Emotion: {emotion}", wrap=True, horizontalalignment='center', fontsize=12)

        plt.tight_layout()

        plt.show()
