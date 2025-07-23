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

from vision import recognize_face, get_lips_landmarks
from post_processing import remove_outliers, stitch_sequences, compute_speaking_probability

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

# Open video
input_path = 'videos/sample5_p.mp4'
cap = cv2.VideoCapture(input_path)

current_faces = []
face_images = []

frames_stack= []
i=0


# Process frames
while True:

    ret, frame = cap.read()
    if not ret:
        break

    if i%5==0:  # Process every 30th frame
        frames_stack.append({})

        # Run YOLO inference
        output = model(frame)
        results = Detections.from_ultralytics(output[0])

        updated_idx = [None] * len(current_faces)

        # Draw bounding boxes
        for detection in results:
            know_face = recognize_face(current_faces, detection[0], frame)
            if know_face is not None:
                current_faces[know_face].append(detection[0])
                updated_idx[know_face] = True
                img_rgb = Image.fromarray(cv2.cvtColor(frame[int(detection[0][1]):int(detection[0][3]), 
                                                                    int(detection[0][0]):int(detection[0][2])], cv2.COLOR_BGR2RGB))
                face_images[know_face] = img_rgb
                frames_stack[i//5].update({know_face: img_rgb})
            else:
                faces = deque(maxlen=LEN_FRAME_BUFFER)
                faces.append(detection[0])
                current_faces.append(faces)
                x1, y1, x2, y2 = detection[0]
                img_rgb = Image.fromarray(cv2.cvtColor(frame[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB))
                face_images.append(img_rgb)
                frames_stack[i//5].update({len(face_images) - 1: img_rgb})

            x1, y1, x2, y2 = detection[0]

        nn_idx = [i for i, updated in enumerate(updated_idx) if updated is None]
        rm_idx = []
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

#build a grid of faces, each row corresponds to a face and each column to a frame
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
            face_grid[i, j] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))  # Placeholder for missing faces

#post-processing
face_grid, sparsity, lips_landmarks_grid = remove_outliers(face_grid, sparsity, lips_landmarks_grid)
face_grid, sparsity, lips_landmarks_grid = stitch_sequences(face_grid, sparsity, lips_landmarks_grid)

# create results folder
output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#save each row in a different subfolder
best_prob = 0
speaking_user = -1
for i in range(face_grid.shape[1]):

    face_row = face_grid[:, i]
    sparsity_row = sparsity[:, i]
    lips_landmarks_row = lips_landmarks_grid[:, i]

    #filter empty frames
    face_row = [face for face, sparse in zip(face_row, sparsity_row) if sparse]
    lips_landmarks_row = [landmark for landmark, sparse in zip(lips_landmarks_row, sparsity_row) if sparse]

    # Create subfolder for each face
    subfolder_path = os.path.join(output_dir, f"{i}")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Save each frame as an image
    for j in range(len(face_row)):
        img = face_row[j]
        img.save(os.path.join(subfolder_path, f'frame_{j:04d}.png'))

    # Compute speaking probability
    # print(lips_landmarks_row)
    probs = compute_speaking_probability(lips_landmarks_row, threshold=1.5)
    mean_prob = np.mean(probs)
    if mean_prob > best_prob:
        best_prob = mean_prob
        speaking_user = i
    print(f"Face {i}: Mean speaking probability = {mean_prob:.2f}")

print("\nProcessing complete. Results saved in 'results' folder.")
print(f"The speaker is the person number {speaking_user}")

# plot the row of the speaking user

import matplotlib.pyplot as plt
speaking_face_row = face_grid[:, speaking_user]
speaking_sparsity_row = sparsity[:, speaking_user]
speaking_face_row = [face for face, sparse in zip(speaking_face_row, speaking_sparsity_row) if sparse]  

n_frames = len(speaking_face_row)
fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2, 2))
for j in range(n_frames):
    ax = axes[j]
    ax.imshow(speaking_face_row[j])
    ax.axis('off')
plt.tight_layout()
plt.show()

