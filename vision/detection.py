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
from config.settings import LEN_FRAME_BUFFER, FRAME_STRIDE

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

# speaking face detection 

def detect_speaking_face(cap, model, landmark_detector, save_frames=False, verbose=True):
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
    frames_stack, current_faces, face_images = [], [], []
    i = 0

    while True:  # Read frames from the video
        ret, frame = cap.read()
        if not ret:
            break

        if i % FRAME_STRIDE == 0:  # Process every FRAME_STRIDE-th frame (for computation efficiency)
            frames_stack.append({})  # Storage for detected faces
            process_detections(frame, model, current_faces, face_images, frames_stack, i)

        i += 1

    # Release the video capture object
    cap.release()

    # Process the frames stack to create a grid of faces (each row corresponds to a frame and each column to a face)
    face_grid, sparsity, lips_landmarks_grid = build_face_grids(frames_stack, landmark_detector)

    # Post-process the grids to remove outliers and stitch sequences
    face_grid, sparsity, lips_landmarks_grid = remove_outliers(face_grid, sparsity, lips_landmarks_grid)
    face_grid, sparsity, lips_landmarks_grid = stitch_sequences(face_grid, sparsity, lips_landmarks_grid)

    # Process the face grid to compute the speaking probability for each face
    speaking_user, speaking_probs = identify_speaking_face(face_grid, sparsity, lips_landmarks_grid, save_frames, verbose)

    # Extract and return speaking face row
    face_row = face_grid[:, speaking_user]
    sparsity_row = sparsity[:, speaking_user]

    return [face for face, sparse in zip(face_row, sparsity_row) if sparse], face_grid, speaking_probs

def process_detections(frame, model, current_faces, face_images, frames_stack, i):
    """
    Process the detections from the YOLO model and update the current faces and frames stack.
    Args:
        frame (numpy.ndarray): Current video frame.
        model: YOLOv8 model for face detection.
        current_faces (list): List of current faces being tracked.
        face_images (list): List of images corresponding to the current faces.
        frames_stack (list): List of frames with detected faces.
        i (int): Current frame index.
    """

    # Run YOLO inference
    output = model(frame)
    results = Detections.from_ultralytics(output[0])
    updated_idx = [None] * len(current_faces)

    for detection in results:
        bbox = detection[0]
        known_face_idx = recognize_face(current_faces, bbox)  # recognize_face function returns index or None
        x1, y1, x2, y2 = map(int, bbox)
        img_rgb = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

        if known_face_idx is not None:
            # If the face is recognized, update the existing face buffer with the new frame
            current_faces[known_face_idx].append(bbox)
            updated_idx[known_face_idx] = True
            face_images[known_face_idx] = img_rgb
            frames_stack[i // FRAME_STRIDE][known_face_idx] = img_rgb  # Update frame stack

        else:
            # If the face is not recognized, create a new face buffer
            faces = deque(maxlen=LEN_FRAME_BUFFER)
            faces.append(bbox)
            current_faces.append(faces)
            updated_idx.append(True)
            face_images.append(img_rgb)
            frames_stack[i // FRAME_STRIDE][len(face_images) - 1] = img_rgb  # Update frame stack

    # Remove faces that have not been updated for a while (we assume they have left the scene)
    remove_stale_faces(current_faces, face_images, updated_idx)

def remove_stale_faces(current_faces, face_images, updated_idx):
    """
    Remove faces that have not been updated for a while.
    """
    # Get the indices of the faces that have not been updated this frame
    nn_idx = [k for k, updated in enumerate(updated_idx) if updated is None]
    for idx in nn_idx:
        if current_faces[idx] is None:
            continue
        current_faces[idx].append(None)  # Indicate this face has not been detected
        if sum(face is None for face in current_faces[idx]) == LEN_FRAME_BUFFER:
            # If face not seen for LEN_FRAME_BUFFER frames, remove it
            current_faces[idx] = None
            face_images[idx] = Image.fromarray(np.zeros(face_images[idx].size, dtype=np.uint8))


def build_face_grids(frames_stack, landmark_detector):
    """
    Build grids of face images, sparsity information, and lips landmarks from the frames stack.
    """
    # Define three different grids to store the face images, sparsity info, and lips landmarks
    keys = np.unique([k for frames in frames_stack if frames for k in frames.keys()])
    n_faces = len(keys)
    n_frames = len(frames_stack)

    face_grid = np.zeros((n_frames, n_faces), dtype=object)
    sparsity = np.zeros((n_frames, n_faces), dtype=bool)  # Indicates if face is detected in a frame
    lips_landmarks_grid = np.zeros((n_frames, n_faces), dtype=object)

    # Fill the grids
    for i, frames in enumerate(frames_stack):
        for j, key in enumerate(keys):
            if key in frames:
                face_grid[i, j] = frames[key]
                sparsity[i, j] = True
                lips_landmarks_grid[i, j] = get_lips_landmarks(np.array(face_grid[i, j]), landmark_detector)
            else:
                face_grid[i, j] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    return face_grid, sparsity, lips_landmarks_grid


def identify_speaking_face(face_grid, sparsity, lips_landmarks_grid, save_frames, verbose=True):
    """
    Identify the speaking face based on the computed speaking probabilities.
    Args:
        face_grid (numpy.ndarray): Grid of face images.
        sparsity (numpy.ndarray): Grid indicating if a face is detected in a frame.
        lips_landmarks_grid (numpy.ndarray): Grid of lips landmarks.
        save_frames (bool): Whether to save frames with detected faces.
        verbose (bool): Whether to print detailed logs.
    Returns:
        int: Index of the speaking face.
        list: List of speaking probabilities for each face.
    """
    probs = []

    for i in range(face_grid.shape[1]):
        face_row = face_grid[:, i]
        sparsity_row = sparsity[:, i]
        lips_row = lips_landmarks_grid[:, i]

        # Keep only frames where face is detected
        valid_faces = [f for f, s in zip(face_row, sparsity_row) if s]
        valid_lips = [l for l, s in zip(lips_row, sparsity_row) if s]

        if save_frames:
            subfolder_path = os.path.join("results", f"{i}")
            os.makedirs(subfolder_path, exist_ok=True)
            for j, img in enumerate(valid_faces):
                img.save(os.path.join(subfolder_path, f'frame_{j:04d}.png'))

        prob_face = compute_speaking_probability(valid_lips)
        mean_prob = np.mean(prob_face)
        if verbose:
            print(f"Face {i}: Mean speaking probability = {mean_prob:.2f}")

        probs.append(mean_prob)

    speaking_user = np.argmax(probs)
    if verbose:
        print("\nProcessing complete. Results saved in 'results' folder.")
        print(f"The speaker is the person number {speaking_user}, with a speaking probability of {probs[speaking_user]:.2f}.")

    return speaking_user, probs