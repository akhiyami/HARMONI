import dlib
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from supervision import Detections
from collections import deque
import sys
import cv2

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from vision_module.post_processing import remove_outliers, stitch_sequences, compute_speaking_probability #TODO: fix path work in all environments

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0

def recognize_face(current_faces, new_face, frame):
    #current_faces list of deque
    for i, faces in enumerate(current_faces):
        if faces is None:
            continue
        last_position = [face for face in faces if face is not None][-1]

        if iou(last_position, new_face) > 0.5:
            return i        
    return None

def get_lips_landmarks(img, landmark_detector):
        
    landmarks = landmark_detector(img, dlib.rectangle(0, 0, img.shape[1], img.shape[0]))
    landmarks_list = [[point.x, point.y] for point in landmarks.parts()[48:68]]  # Get the landmarks for the mouth (indices 48 to 67)

    rotated_ld = rotate_landmarks_horizontal(landmarks_list)
    centered_ld = center_landmarks(rotated_ld)
    
    return centered_ld
        
def rotate_landmarks_horizontal(landmarks):
    """
    Rotates landmarks so that the line between landmark 0 and 6 is horizontal.
    The rotation is around landmark 0, and landmark 0 will be at (0, 0).
    """
    landmarks = np.array(landmarks)
    anchor = landmarks[0]
    target = landmarks[6]
    
    # Vector from landmark 0 to landmark 6
    vec = target - anchor
    dx, dy = vec

    # Compute angle to horizontal
    angle = -np.arctan2(dy, dx)  # negative to rotate clockwise

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    # Translate landmarks to make landmark 0 the origin
    translated = landmarks - anchor

    # Apply rotation
    rotated = translated @ rotation_matrix.T  # [N,2] x [2,2]

    return rotated

def center_landmarks(landmarks):
    """
    Translate landmarks so the abscissa of landmark 3 is 0
    """
    landmarks = np.array(landmarks)
    landmark_3 = landmarks[3]
    centered_landmarks = landmarks - [landmark_3[0], 0]
    return centered_landmarks


def detect_speaking_face(cap, model, landmark_detector, save_frames=False):
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

    if save_frames:
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
        
        if save_frames:
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