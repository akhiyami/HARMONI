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
from insightface.app import FaceAnalysis
from scipy.signal import hilbert, correlate

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid GUI warnings
import matplotlib.pyplot as plt

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from vision.utils import remove_outliers, stitch_sequences
from vision.audio import extract_audio, get_waveform
from config.settings import LEN_FRAME_BUFFER, FRAME_STRIDE

app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(640, 640))

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
    
    return landmarks_list
        

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

def detect_speaking_face(video_path, model, insightface_model, save_frames=False, verbose=True):
    """
    Detect speaking faces in a video stream using YOLOv8 and Dlib.
    Args:
        video_path: Path to the video file.
        model: YOLOv8 model for face detection.
        landmark_detector: Dlib landmark detector for lip detection.
        save_frames (bool): Whether to save frames with detected faces.
    Returns:
        list: List of detected speaking faces.
    """
    frames_stack, current_faces, face_images = [], [], []
    i = 0

    cap = cv2.VideoCapture(video_path)

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

    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    
    waveform_plot = get_waveform(audio_path)

    # Process the frames stack to create a grid of faces (each row corresponds to a frame and each column to a face)
    face_grid, sparsity, landmarks_grid = build_face_grids(frames_stack, None)

    # Post-process the grids to remove outliers and stitch sequences
    face_grid, sparsity, landmarks_grid = remove_outliers(face_grid, sparsity, landmarks_grid)
    face_grid, sparsity, landmarks_grid = stitch_sequences(face_grid, sparsity, landmarks_grid)

    # Process the face grid to compute the speaking probability for each face
    speaking_user, speaking_probs = identify_speaking_face(face_grid, sparsity, landmarks_grid, save_frames, waveform_plot, verbose)

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
    # Initialize InsightFace app for face detection
    faces = app.get(frame)

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
    # Collect all unique face indices (keys) across frames_stack
    keys = np.unique([k for frames in frames_stack if frames for k in frames.keys()])

    # frames_stack[frame_idx][face_idx] = (img_rgb, landmarks)

    n_faces = len(keys)
    n_frames = len(frames_stack)

    face_grid = np.zeros((n_frames, n_faces), dtype=object)
    sparsity = np.zeros((n_frames, n_faces), dtype=bool)  # Indicates if face is detected in a frame
    landmarks_grid = np.zeros((n_frames, n_faces), dtype=object)

    # Fill the grids
    for i, frames in enumerate(frames_stack):
        for j, key in enumerate(keys):
            if key in frames:
                sparsity[i, j] = True
                # Draw landmarks on the face image
                face_img, landmarks = np.array(frames[key][0]).copy(), frames[key][1]
                lip_landmarks = landmarks[52:72] # Get the landmarks for the mouth (indices 52 to 71)
                right_eye_landmarks = landmarks[39]  
                left_eye_landmarks = landmarks[89]
                # landmarks = get_lips_landmarks(face_img, landmark_detector)
                for (x, y) in lip_landmarks:
                    cv2.circle(face_img, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv2.circle(face_img, (int(right_eye_landmarks[0]), int(right_eye_landmarks[1])), 2, (255, 0, 0), -1)
                cv2.circle(face_img, (int(left_eye_landmarks[0]), int(left_eye_landmarks[1])), 2, (255, 0, 0), -1)
                face_grid[i, j] = Image.fromarray(face_img)
                landmarks_grid[i, j] = landmarks

            else:
                face_grid[i, j] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    return face_grid, sparsity, landmarks_grid

def compute_speaking_probability(landmark_sequences, smooth_window=3, threshold=0.02, face_id=None, waveform_plot=None):
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

    distances_all = []
    for t in range(T):
        ld = landmarks[t]

        # distance between the eyes
        eye_dist = np.linalg.norm(ld[39] - ld[89])  # Distance between eyes for normalization 

        # inner distances between lips
        D_in_c = np.linalg.norm(ld[62] - ld[60])   # center of the mouth
        D_in_l = np.linalg.norm(ld[66] - ld[54])   # left side of the mouth
        D_in_r = np.linalg.norm(ld[70] - ld[57])   # right side of the mouth

        # outer distances between lips
        D_out_c = np.linalg.norm(ld[71] - ld[53])  # center of the mouth
        D_out_l = np.linalg.norm(ld[63] - ld[56])  # left side of the mouth
        D_out_r = np.linalg.norm(ld[67] - ld[59])  # right side of the mouth

        A = (D_in_c + D_in_r + D_in_l) / 3.0  # Average inner distance
        B = (D_out_c + D_out_r + D_out_l) / 3.0  # Average outer distance

        #distance between mouth corners
        d_corners = np.linalg.norm(ld[52] - ld[61])
        normalized_d_corners = d_corners / (eye_dist + 1e-6)

        alpha = 0.8 # Weighting factor between inner and outer distances

        distances_all.append({"d_corners": d_corners, "d_eyes": eye_dist, "normalized_d_corners": normalized_d_corners,
                              "A": A, "B": B, "num":(alpha * A + (1-alpha) * B)})

        mar = (alpha * A + (1-alpha) * B) / (d_corners + 1e-6)
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

    waveform_axis, waveform = waveform_plot
    stride = len(waveform) // len(mar_values)

    analytic_signal = hilbert(waveform)
    envelope = np.abs(analytic_signal)
    sound_variations = envelope[::stride][:len(mar_variations)]

    #normalize waveform to [0, 1]
    sound_variations = (sound_variations - np.min(sound_variations)) / (np.max(sound_variations) - np.min(sound_variations) + 1e-6)
    #normalize mar_variations to [0, 1]
    mar_variations = (mar_variations - np.min(mar_variations)) / (np.max(mar_variations) - np.min(mar_variations) + 1e-6)

    plt.figure(figsize=(10, 3))
    plt.plot(sound_variations, label='Waveform', color='gray')
    plt.plot(mar_variations, label='MAR variations', color='orange')
    plt.title("Waveform and MAR variations over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(f"waveform_face_{face_id}.png")
    plt.close()

    #copute correlation between waveform and mar_variations
    corr = correlate(mar_variations - np.mean(mar_variations), sound_variations - np.mean(sound_variations), mode='full')
    print(corr)

    print(f"Face {face_id}: Correlation between MAR variations and waveform = {np.max(corr):.2f}")


    # Convert to probability (sigmoid)
    probs = 1 / (1 + np.exp(- (mar_variations - threshold) * 50))

    return probs.tolist()

def identify_speaking_face(face_grid, sparsity, landmarks_grid, save_frames, waveform_plot=None, verbose=True):
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

        prob_face = compute_speaking_probability(valid_landmarks, face_id=i, waveform_plot=waveform_plot)
        mean_prob = np.mean(prob_face)
        if verbose:
            print(f"Face {i}: Mean speaking probability = {mean_prob:.2f}")

        probs.append(mean_prob)

    speaking_user = np.argmax(probs)
    if verbose:
        print("\nProcessing complete. Results saved in 'results' folder.")
        print(f"The speaker is the person number {speaking_user}, with a speaking probability of {probs[speaking_user]:.2f}.")

    return speaking_user, probs