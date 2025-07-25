from ultralytics import YOLO
import cv2
from huggingface_hub import hf_hub_download
from supervision import Detections
import numpy as np
from PIL import Image
import torch
from collections import deque
import dlib
import cv2
import os
import matplotlib.pyplot as plt
import whisper
from concurrent.futures import ThreadPoolExecutor
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import AutoImageProcessor, AutoModelForImageClassification
from dotenv import load_dotenv

from vision import recognize_face, get_lips_landmarks
from post_processing import remove_outliers, stitch_sequences, compute_speaking_probability
from audio import transcribe_audio, extract_audio
from emotions import generate_description, detect_emotions

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# Define the name of the video to process
name_video = 'sample4'  # Change this to your video name without extension

#empty result folder if exists
output_dir = 'results'
if os.path.exists(output_dir):
    #remove the folder directly executing command line `rm -rf results`
    os.system(f'rm -rf {output_dir}')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
LEN_FRAME_BUFFER = 15

# Load YOLOv8 model for face detection
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# load landmark detector
landmark_detector = dlib.shape_predictor(os.path.join('./models', 'shape_predictor_68_face_landmarks.dat'))

#load whisper model
model_size = "base"  # or "small", "medium", "large"
whisper_model = whisper.load_model(model_size)

# Open video
input_path = f'videos/{name_video}.mp4'
cap = cv2.VideoCapture(input_path)

# Model for image description
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# model_id = "ACIDE/User-VLM-10B-base"
# description_processor = PaliGemmaProcessor.from_pretrained(model_id)
# description_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Model for emotion detection
emotion_processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
emotion_model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

# Process frames
def detect_speaking_face(cap):
    frames_stack = []
    current_faces = []
    face_images = []
    i = 0
    LEN_FRAME_BUFFER = 15  # adjust as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % 5 == 0:
            frames_stack.append({})

            # Run YOLO inference
            output = model(frame)
            results = Detections.from_ultralytics(output[0])
            updated_idx = [None] * len(current_faces)

            for detection in results:
                know_face = recognize_face(current_faces, detection[0], frame)
                if know_face is not None:
                    current_faces[know_face].append(detection[0])
                    updated_idx[know_face] = True
                    x1, y1, x2, y2 = detection[0]
                    img_rgb = Image.fromarray(cv2.cvtColor(frame[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB))
                    face_images[know_face] = img_rgb
                    frames_stack[i // 5].update({know_face: img_rgb})
                else:
                    faces = deque(maxlen=LEN_FRAME_BUFFER)
                    faces.append(detection[0])
                    current_faces.append(faces)
                    x1, y1, x2, y2 = detection[0]
                    img_rgb = Image.fromarray(cv2.cvtColor(frame[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB))
                    face_images.append(img_rgb)
                    frames_stack[i // 5].update({len(face_images) - 1: img_rgb})

            nn_idx = [k for k, updated in enumerate(updated_idx) if updated is None]
            for idx in nn_idx:
                if current_faces[idx] is None:
                    continue
                current_faces[idx].append(None)
                if sum(face is None for face in current_faces[idx]) == LEN_FRAME_BUFFER:
                    current_faces[idx] = None
                    face_images[idx] = Image.fromarray(np.zeros(face_images[idx].size, dtype=np.uint8))

        i += 1

    cap.release()
    print("End video with face detections.")

    keys = np.unique([k for frames in frames_stack if frames for k in frames.keys()])
    n_faces = len(keys)
    n_frames = len(frames_stack)

    face_grid = np.zeros((n_frames, n_faces), dtype=object)
    sparsity = np.zeros((n_frames, n_faces), dtype=bool)
    lips_landmarks_grid = np.zeros((n_frames, n_faces), dtype=object)

    for i, frames in enumerate(frames_stack):
        for j, key in enumerate(keys):
            if key in frames:
                face_grid[i, j] = frames[key]
                sparsity[i, j] = True
                lips_landmarks_grid[i, j] = get_lips_landmarks(np.array(face_grid[i, j]), landmark_detector)
            else:
                face_grid[i, j] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    face_grid, sparsity, lips_landmarks_grid = remove_outliers(face_grid, sparsity, lips_landmarks_grid)
    face_grid, sparsity, lips_landmarks_grid = stitch_sequences(face_grid, sparsity, lips_landmarks_grid)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    best_prob = 0
    speaking_user = -1
    for i in range(face_grid.shape[1]):
        face_row = face_grid[:, i]
        sparsity_row = sparsity[:, i]
        lips_landmarks_row = lips_landmarks_grid[:, i]

        face_row = [face for face, sparse in zip(face_row, sparsity_row) if sparse]
        lips_landmarks_row = [landmark for landmark, sparse in zip(lips_landmarks_row, sparsity_row) if sparse]

        subfolder_path = os.path.join(output_dir, f"{i}")
        os.makedirs(subfolder_path, exist_ok=True)

        for j in range(len(face_row)):
            img = face_row[j]
            img.save(os.path.join(subfolder_path, f'frame_{j:04d}.png'))

        probs = compute_speaking_probability(lips_landmarks_row)
        mean_prob = np.mean(probs)
        if mean_prob > best_prob:
            best_prob = mean_prob
            speaking_user = i
        print(f"Face {i}: Mean speaking probability = {mean_prob:.2f}")

    print("\nProcessing complete. Results saved in 'results' folder.")
    print(f"The speaker is the person number {speaking_user}")

    # Extract and return speaking face row
    speaking_face_row = face_grid[:, speaking_user]
    speaking_sparsity_row = sparsity[:, speaking_user]
    speaking_face_row = [face for face, sparse in zip(speaking_face_row, speaking_sparsity_row) if sparse]
    
    return speaking_face_row

def extract_and_transcribe_audio(video_file):
    audio_file = "temp_audio.wav"
    extract_audio(video_file, audio_file)
    transcript = transcribe_audio(audio_file, whisper_model)
    os.remove(audio_file)
    return transcript

if __name__ == "__main__":

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_face = executor.submit(detect_speaking_face, cap)
        future_transcript = executor.submit(extract_and_transcribe_audio, f'videos/{name_video}.mp4')

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


