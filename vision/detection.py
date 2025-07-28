"""
This module provides functionality for face detection and speaker recognition in video frames.
"""

#--------------------------------------- Imports ---------------------------------------#

import dlib
import numpy as np
import os
from PIL import Image
from supervision import Detections
from collections import deque
import sys
import cv2

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from vision.post_processing import remove_outliers, stitch_sequences, compute_speaking_probability 
from vision.config.settings import LEN_FRAME_BUFFER, FRAME_STRIDE

#--------------------------------------- Functions ---------------------------------------#

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1 (list): Bounding box in the format [x1, y1, x2, y2].
        box2 (list): Bounding box in the format [x1, y1, x2, y2].
    Returns:
        float: IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0

def recognize_face(current_faces, new_face):
    """
    Recognize a new face by comparing it with existing faces in the current_faces list.
    Args:
        current_faces (list): List of current faces, each face is a deque of bounding boxes.
        new_face (list): New face bounding box in the format [x1, y1, x2, y2].
    Returns:
        int: Index of the recognized face in current_faces, or None if not recognized.
    """
    #current_faces list of deque
    for i, faces in enumerate(current_faces):
        if faces is None:
            continue
        last_position = [face for face in faces if face is not None][-1]

        if iou(last_position, new_face) > 0.5:
            return i        
    return None

def get_lips_landmarks(img, landmark_detector):
    """
    Get the landmarks for the lips from a face image.
    Args:
        img (numpy.ndarray): Face image in RGB format.
        landmark_detector: Dlib landmark detector.
    Returns:
        list: List of landmarks for the lips.
    """
    
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
    """
    Detect speaking faces in a video stream using YOLOv8 and Dlib.
    Args:
        cap: Video capture object.
        model: YOLOv8 model for face detection.
        landmark_detector: Dlib landmark detector for lip detection.
        save_frames (bool): Whether to save frames with detected faces.
    Returns:
        list: List of detected speaking faces.
    """
    frames_stack = []
    current_faces = []
    face_images = []
    i = 0

    while True: # Read frames from the video
        ret, frame = cap.read()
        if not ret:
            break

        if i % FRAME_STRIDE == 0: # Process every FRAME_STRIDE-th frame (for computation efficiency)

            # Storage for detected faces 
            frames_stack.append({}) 

            # Run YOLO inference
            output = model(frame)
            results = Detections.from_ultralytics(output[0])

            updated_idx = [None] * len(current_faces)

            # Check or the detected faces if they are already known
            for detection in results:
                know_face = recognize_face(current_faces, detection[0], frame) # recognize_face function returns the index of the known face or None if not recognized
                if know_face is not None:
                    # If the face is recognized, update the existing face buffer with the new frame, and indicate that this face has been updated
                    current_faces[know_face].append(detection[0])
                    updated_idx[know_face] = True

                    # Extract the face image and convert it to RGB
                    x1, y1, x2, y2 = detection[0]
                    img_rgb = Image.fromarray(cv2.cvtColor(frame[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB))
                    face_images[know_face] = img_rgb

                    # Update the frames stack with the known face, and the associated index corresponding to the identified face
                    frames_stack[i // FRAME_STRIDE].update({know_face: img_rgb})

                else: # If the face is not recognized, create a new face buffer 

                    faces = deque(maxlen=LEN_FRAME_BUFFER)
                    faces.append(detection[0])
                    current_faces.append(faces)

                    # Add the new face to the face images list
                    x1, y1, x2, y2 = detection[0]
                    img_rgb = Image.fromarray(cv2.cvtColor(frame[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB))
                    face_images.append(img_rgb)

                    # Update the frames stack with the new face and its index
                    frames_stack[i // FRAME_STRIDE].update({len(face_images) - 1: img_rgb})

            # Remove faces that have not been updated for a while (we assume that the identified face has left the scene)
            nn_idx = [k for k, updated in enumerate(updated_idx) if updated is None] # Get the indices of the faces that have not been updated this frame
            for idx in nn_idx:
                if current_faces[idx] is None: # this face has already been removed
                    continue

                current_faces[idx].append(None)  # Append None to the deque to indicate that this face has not been detected in this frame
                if sum(face is None for face in current_faces[idx]) == LEN_FRAME_BUFFER: # If the face has not been detected for LEN_FRAME_BUFFER frames, remove it
                    current_faces[idx] = None
                    face_images[idx] = Image.fromarray(np.zeros(face_images[idx].size, dtype=np.uint8))
        i += 1

    # Release the video capture object
    cap.release()

    # Process the frames stack to create a grid of faces (each row corresponds to a frame and each column to a face))
    keys = np.unique([k for frames in frames_stack if frames for k in frames.keys()])
    n_faces = len(keys)
    n_frames = len(frames_stack)

    # define three different grids to store the face images, sparsity information, and lips landmarks
    face_grid = np.zeros((n_frames, n_faces), dtype=object)
    sparsity = np.zeros((n_frames, n_faces), dtype=bool) # boolean array to indicate if the face has been detected in the frame
    lips_landmarks_grid = np.zeros((n_frames, n_faces), dtype=object)

    # Fill the grids with the face images, sparsity information, and lips landmarks
    for i, frames in enumerate(frames_stack):
        for j, key in enumerate(keys):
            if key in frames:
                face_grid[i, j] = frames[key]
                sparsity[i, j] = True
                lips_landmarks_grid[i, j] = get_lips_landmarks(np.array(face_grid[i, j]), landmark_detector)
            else:
                face_grid[i, j] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    # Post-process the grids to remove outliers and stitch sequences
    face_grid, sparsity, lips_landmarks_grid = remove_outliers(face_grid, sparsity, lips_landmarks_grid)
    face_grid, sparsity, lips_landmarks_grid = stitch_sequences(face_grid, sparsity, lips_landmarks_grid)

    # Save the frames if required
    if save_frames:
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)


    # Process the face grid to compute the speaking probability for each face
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