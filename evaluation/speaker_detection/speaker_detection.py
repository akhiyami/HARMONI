import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sqlite3
import time

import cv2

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_folder_path)

from vision.detection import detect_speaking_face
from conversation.memory.memory import user_retriever

from config import get_face_embedding_model

user_retriever_config = get_face_embedding_model("INSIGHTFACE")

videos_path = "data/broca-scenarios"

database = 'evaluation/users.db'
conn = sqlite3.connect(database)

videos = os.listdir(videos_path)
videos = [v for v in videos if v.endswith('.mp4')]

results_dir = "results/speaker_identification"
os.makedirs(results_dir, exist_ok=True)


detect_face_time = 0
user_retrieval_time = 0
speaker_id_time = 0

for video in tqdm(videos):
    print(f"Processing video: {video}")

    #get time of a video
    video_duration = 0
    video_capture = cv2.VideoCapture(os.path.join(videos_path, video))
    if video_capture.isOpened():
        video_duration = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / video_capture.get(cv2.CAP_PROP_FPS)
        n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()

    start_time = time.time()
    speaking_face_row, grid, probs, speaker_time = detect_speaking_face(video_path=os.path.join(videos_path, video), save_frames=False, verbose=False)

    detect_face_time += (time.time() - start_time - speaker_time) / n_frames
    speaker_id_time += speaker_time 
    best_idx = np.argmax(probs)

    result_dir_video = os.path.join(results_dir, video[:-4])
    os.makedirs(result_dir_video, exist_ok=True)

    user_retrieval_mean_time = 0
    for face in range(grid.shape[1]):

        face_row = grid[:, face]
        # Find the first non-all-black image in face_row
        image = None
        for img in face_row:
            if np.any(img):  # Checks if there is any non-zero pixel
                image = img
        if image is None:
            image = face_row[0]  # fallback to the first image if all are black

        #detect_user

        start_time = time.time()
        detected_user, _ = user_retriever(image, conn, user_retriever_config, database= database)
        user_retrieval_mean_time += time.time() - start_time

        plt.imshow(image)
        plt.axis('off')
        if face == best_idx:
            plt.title(f"Prob: {probs[face]:.2f} (detected speaker) - User: {detected_user}")
            plt.savefig(os.path.join(result_dir_video, f"face_{face}_detected_speaker.png"))
        else:
            plt.title(f"Prob: {probs[face]:.2f} - User: {detected_user}")
            plt.savefig(os.path.join(result_dir_video, f"face_{face}.png"))
        plt.close()

    user_retrieval_mean_time /= grid.shape[1]
    user_retrieval_time += user_retrieval_mean_time 
    print(f"Detected speaking face: {best_idx} with probability {probs[best_idx]:.2f}")
    #save original video
    os.system(f"cp {os.path.join(videos_path, video)} {os.path.join(result_dir_video, video)}")

user_retrieval_time /= len(videos)
detect_face_time /= len(videos)
speaker_id_time /= len(videos)

print(f"Average time to detect speaking face: {detect_face_time:.4f} seconds")
print(f"Average time to retrieve user: {user_retrieval_time:.4f} seconds")
print(f"Average time for speaker identification: {speaker_id_time:.4f} seconds")

