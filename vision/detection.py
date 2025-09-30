"""
This module provides functionality for face detection and speaker recognition in video frames.
"""

#--------------------------------------- Imports ---------------------------------------#

import os
import sys
import time
from collections import deque

import cv2
import dlib
import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis
from scipy.signal import hilbert, correlate

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid GUI warnings
import matplotlib.pyplot as plt

from config.utils import suppress_stdout
from config.settings import LEN_FRAME_BUFFER, FRAME_STRIDE

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from vision.utils import remove_outliers, stitch_sequences, recognize_face, remove_stale_faces, build_face_grids
from vision.audio import get_vad_segments, vad_flags_for_frames, extract_mono_wav

#--------------------------------------- Initialization ---------------------------------------#

with suppress_stdout():
    FACE_APP = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    FACE_APP.prepare(ctx_id=0, det_size=(640, 640))

#--------------------------------------- Functions ---------------------------------------#
        
# speaking face detection 

def detect_speaking_face(video_path, save_frames=False, verbose=True):
    """
    Detect speaking faces in a video stream using YOLOv8 and Dlib.
    Args:
        video_path: Path to the video file.
        save_frames (bool): Whether to save frames with detected faces.
    Returns:
        list: List of detected speaking faces.
    """
    frames_stack, current_faces, face_images = [], [], []
    i = 0

    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:  # Read frames from the video
        ret, frame = cap.read()
        if not ret:
            break

        if i % FRAME_STRIDE == 0:  # Process every FRAME_STRIDE-th frame (for computation efficiency)
            frames_stack.append({})  # Storage for detected faces
            process_detections(frame, current_faces, face_images, frames_stack, i)

        i += 1

    # Release the video capture object
    cap.release()

    audio_path = "temp_audio.wav"
    audio_path = extract_mono_wav(video_path, wav_path=audio_path)

    # --- run VAD ---
    vad_start_time = time.time()
    segments = get_vad_segments(audio_path)

    # --- align with frames ---
    vad_flags = vad_flags_for_frames(segments, num_frames=len(frames_stack),
                                    fps=fps, frame_stride=FRAME_STRIDE)
    vad_time = time.time() - vad_start_time

    # Process the frames stack to create a grid of faces (each row corresponds to a frame and each column to a face)
    face_grid, sparsity, landmarks_grid = build_face_grids(frames_stack)

    # Post-process the grids to remove outliers and stitch sequences
    face_grid, sparsity, landmarks_grid = remove_outliers(face_grid, sparsity, landmarks_grid)
    face_grid, sparsity, landmarks_grid = stitch_sequences(face_grid, sparsity, landmarks_grid)

    # Process the face grid to compute the speaking probability for each face
    speaker_start_time = time.time()
    speaking_user, speaking_probs = identify_speaking_face(face_grid, sparsity, landmarks_grid, save_frames, vad_flags, verbose)
    speaker_time = time.time() - speaker_start_time

    process_video_time = time.time() - start_time

    # Extract and return speaking face row
    face_row = face_grid[:, speaking_user]
    sparsity_row = sparsity[:, speaking_user]

    return [face for face, sparse in zip(face_row, sparsity_row) if sparse], face_grid, speaking_probs

def process_detections(frame, current_faces, face_images, frames_stack, i):
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
    # Initialize InsightFace app for face detection
    faces = FACE_APP.get(frame)

    # # Run YOLO inference
    # output = model(frame)
    # results = Detections.from_ultralytics(output[0])

    updated_idx = [None] * len(current_faces)

    for detection in faces: #results if yolo
        # bbox = detection[0]
        bbox = detection['bbox'] # InsightFace bbox format: [x1, y1, x2, y2]
        landmarks = np.round(detection['landmark_2d_106']).astype(np.int32)
        known_face_idx = recognize_face(current_faces, bbox)  # recognize_face function returns index or None
        x1, y1, x2, y2 = map(int, bbox)
        # Clip coordinates to image size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        # Skip if the bbox is invalid
        if x2 <= x1 or y2 <= y1:
            continue
        
        img_rgb = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        landmarks = landmarks - [x1, y1]  # Adjust landmarks to be relative to the cropped face image

        if known_face_idx is not None:
            # If the face is recognized, update the existing face buffer with the new frame
            current_faces[known_face_idx].append(bbox)
            updated_idx[known_face_idx] = True
            face_images[known_face_idx] = img_rgb
            frames_stack[i // FRAME_STRIDE][known_face_idx] = (img_rgb, landmarks)  # Update frame stack with image and landmarks

        else:
            # If the face is not recognized, create a new face buffer
            faces = deque(maxlen=LEN_FRAME_BUFFER)
            faces.append(bbox)
            current_faces.append(faces)
            updated_idx.append(True)
            face_images.append(img_rgb)
            frames_stack[i // FRAME_STRIDE][len(face_images) - 1] = (img_rgb, landmarks)  # Update frame stack with image and landmarks

    # Remove faces that have not been updated for a while (we assume they have left the scene)
    remove_stale_faces(current_faces, face_images, updated_idx)


def compute_speaking_probability(landmark_sequences, smooth_window=3, threshold=0.02, face_id=None, vad_flags=None):
    """
    Estimate speaking probability using MAR, assuming only mouth landmarks [48:68] are passed.

    Parameters:
        landmark_sequences: np.ndarray of shape [T, 106, 2]
            - T = number of frames
            - 106 facial landmarks (indices correspond to 0–105 in insightface)
        smooth_window: int
            - Rolling window size for temporal smoothing
        threshold: float
            - Sensitivity for MAR change above baseline

    Returns:
        probs: list of floats, speaking probability per frame
    """
    landmarks = np.array(landmark_sequences)  # [T, 106, 2]
    T = landmarks.shape[0]

    mar_values = []

    for t in range(T):
        ld = landmarks[t]

        # distance between the eyes
        eye_dist = np.linalg.norm(ld[39] - ld[89])  # Distance between eyes for normalization 

        # inner distances between lips
        D_in_c = np.linalg.norm(ld[62] - ld[60])   # center of the mouth
        D_in_l = np.linalg.norm(ld[66] - ld[54])   # left side of the mouth
        D_in_r = np.linalg.norm(ld[70] - ld[57])   # right side of the mouth

        A = (D_in_c + D_in_r + D_in_l) / 3.0  # Average inner distance

        #distance between mouth corners
        d_corners = np.linalg.norm(ld[52] - ld[61])
        normalized_d_corners = d_corners / (eye_dist + 1e-6)

        mar = A / (normalized_d_corners + 1e-6)
        mar_values.append(mar)

    mar_values = np.array(mar_values)    

    # Smooth over time
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window 
        mar_smooth = np.convolve(mar_values, kernel, mode='same')
    else:
        mar_smooth = mar_values

    # Estimate baseline (closed mouth MAR)
    baseline = np.percentile(mar_smooth, 10)
    mar_delta = mar_smooth - baseline
    mar_variations = np.abs(np.diff(mar_smooth, prepend=mar_smooth[0]))

    # Step 1: feature likelihoods
    p_vis = 1 / (1 + np.exp(-(mar_delta / threshold)))

    ####
    # Pupil visibility
    ####
    right_open, left_open = [], []
    right_gaze, left_gaze = [], []

    for t in range(T):
        ld = landmarks[t]

        # --- Right eye ---
        top_r, bottom_r = ld[94], ld[87]
        left_r, right_r = ld[89], ld[93]
        pupil_r = ld[88]
        width_r = np.linalg.norm(left_r - right_r)
        open_r = np.linalg.norm(top_r - bottom_r) / (width_r + 1e-6)
        center_r = (left_r + right_r) / 2
        gaze_r = np.linalg.norm(pupil_r - center_r) / (width_r + 1e-6)

        # --- Left eye ---
        top_l, bottom_l = ld[40], ld[33]
        left_l, right_l = ld[35], ld[39]
        pupil_l = ld[38]
        width_l = np.linalg.norm(left_l - right_l)
        open_l = np.linalg.norm(top_l - bottom_l) / (width_l + 1e-6)
        center_l = (left_l + right_l) / 2
        gaze_l = np.linalg.norm(pupil_l - center_l) / (width_l + 1e-6)

        right_open.append(open_r)
        left_open.append(open_l)
        right_gaze.append(gaze_r)
        left_gaze.append(gaze_l)

    # Convert to numpy
    right_open, left_open = np.array(right_open), np.array(left_open)
    right_gaze, left_gaze = np.array(right_gaze), np.array(left_gaze)

    # Eye openness (probability eye is open)
    eye_open = (right_open + left_open) / 2
    p_open = 1 / (1 + np.exp(-(eye_open - 0.25) * 20))  # threshold ~0.25

    # Gaze alignment (probability looking at center)
    gaze_offset = (right_gaze + left_gaze) / 2
    p_gaze = np.exp(-(gaze_offset**2) / (0.05**2))  # Gaussian around 0

    # Final probability: open eyes AND looking at center
    p_pupil = p_open * (1 - p_gaze)

    # Combine probabilities
    alpha = 0.75
    prob = alpha * p_vis + (1 - alpha) * p_pupil

    #keep prob only when VAD says speech
    if vad_flags is not None and len(vad_flags) == len(prob):
        prob = [p for p, vad in zip(prob, vad_flags) if vad]
        p_vis = [p for p, vad in zip(p_vis, vad_flags) if vad]
        p_pupil = [p for p, vad in zip(p_pupil, vad_flags) if vad]

    return np.mean(prob)


def identify_speaking_face(face_grid, sparsity, landmarks_grid, save_frames, vad_flags=None, verbose=True):
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
        landmarks_row = landmarks_grid[:, i]

        # Keep only frames where face is detected
        valid_faces = [f for f, s in zip(face_row, sparsity_row) if s]
        valid_landmarks = [l for l, s in zip(landmarks_row, sparsity_row) if s]

        if save_frames:
            subfolder_path = os.path.join("results", f"{i}")
            os.makedirs(subfolder_path, exist_ok=True)
            for j, img in enumerate(valid_faces):
                img.save(os.path.join(subfolder_path, f'frame_{j:04d}.png'))

        prob_face = compute_speaking_probability(valid_landmarks, face_id=i, vad_flags=vad_flags)
        # prob_face = talking_to_camera_probability(valid_landmarks)
        # mean_prob = np.mean(prob_face) CHANGE PROB_FACE IF USING MEAN
        if verbose:
            print(f"Face {i}: Mean speaking probability = {prob_face:.2f}")

        probs.append(prob_face)

    speaking_user = np.argmax(probs)
    if verbose:
        print("\nProcessing complete. Results saved in 'results' folder.")
        print(f"The speaker is the person number {speaking_user}, with a speaking probability of {probs[speaking_user]:.2f}.")

    return speaking_user, probs


def detect_faces_image(image):
    """
    Detect faces in a single image using InsightFace.
    Args:
        image (PIL.Image or numpy.ndarray): Input image.
    Returns:
        list: List of bounding boxes for detected faces.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    faces = FACE_APP.get(image)

    bboxes = []
    for face in faces:
        bbox = face['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        bboxes.append([x1, y1, x2, y2])

    return bboxes