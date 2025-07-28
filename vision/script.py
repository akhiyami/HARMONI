"""
Script to process a video, detect speaking faces, transcribe audio, and analyze emotions.
"""

#--------------------------------------- Imports ---------------------------------------#

import cv2
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from vision.detection import detect_speaking_face
from vision.audio import extract_and_transcribe_audio
from vision.emotions import detect_emotions
from vision.config import models
from vision.config.settings import LEN_FRAME_BUFFER, HF_TOKEN

#--------------------------------------- Configuration ---------------------------------------#

# Define the name of the video to process
name_video = 'sample4'  # Change this to your video name without extension

# Load models
model = models.YOLO_FACE_MODEL
landmark_detector = models.LANDMARK_DETECTOR
whisper_model = models.WHISPER_MODEL
emotion_model = models.EMOTION_MODEL
emotion_processor = models.EMOTION_PROCESSOR

#--------------------------------------- Main Execution ---------------------------------------#

if __name__ == "__main__":

    # Empty result folder if exists
    output_dir = 'results'
    if os.path.exists(output_dir):
        #remove the folder directly executing command line `rm -rf results`
        os.system(f'rm -rf {output_dir}')

    # Open video
    input_path = f'videos/{name_video}.mp4'
    cap = cv2.VideoCapture(input_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_face = executor.submit(detect_speaking_face, cap, model, landmark_detector, save_frames=True)
        future_transcript = executor.submit(extract_and_transcribe_audio, f'videos/{name_video}.mp4', whisper_model)

        speaking_face_row = future_face.result()
        transcript = future_transcript.result()

    #pick 10 frames equally spaced
    if len(speaking_face_row) > 10:
        speaking_face_row_array = np.array(speaking_face_row, dtype=object)
        sample_speaking_face_row = speaking_face_row_array[np.linspace(0, len(speaking_face_row_array) - 1, 10, dtype=int)].tolist()
    else:
        sample_speaking_face_row = speaking_face_row
    

    import time
    start_time = time.time()
    emotion, prob = detect_emotions(speaking_face_row, emotion_model, emotion_processor)
    end_time = time.time()
    print(f"Emotion generated in {end_time - start_time:.2f} seconds")
    print(f"Detected emotion: {emotion} with probability: {prob}")

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


