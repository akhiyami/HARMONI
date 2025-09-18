"""
Post-processing functions for detected faces grids.
"""

#--------------------------------------- Imports ---------------------------------------#

import numpy as np
from PIL import Image

from config.settings import FILTER_THRESHOLD, LEN_FRAME_BUFFER

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

def remove_outliers(grid, sparsity, lips_landmarks_grid):
    """
    Remove outliers from the detected faces grids based on sparsity.
    """
    sparsity_rows = np.sum(sparsity, axis=0) 

    ignored_rows = np.where(sparsity_rows < FILTER_THRESHOLD)[0]
    if ignored_rows.size > 0:
        face_grid_clean = np.delete(grid, ignored_rows, axis=1)
        sparsity_clean = np.delete(sparsity, ignored_rows, axis=1)
        lips_landmarks_grid_clean = np.delete(lips_landmarks_grid, ignored_rows, axis=1)
    else:
        face_grid_clean = grid
        sparsity_clean = sparsity
        lips_landmarks_grid_clean = lips_landmarks_grid

    return face_grid_clean, sparsity_clean, lips_landmarks_grid_clean

def stitch_sequences(grid, sparsity, lips_landmarks_grid):
    """
    Stitch sequences of detected faces grids based on sparsity.
    This function merges consecutive rows in the grid if they are not separated by a significant gap in sparsity.
    """
    stitched_face_grid = grid.copy()
    stitched_sparsity = sparsity.copy()
    stitched_lips_landmarks_grid = lips_landmarks_grid.copy()

    start_end_list = np.array([(min(np.where(line)[0]), max(np.where(line)[0])) for line in sparsity.T if np.any(line)])
    n_frames, n_rows = grid.shape
    current_row = 0
    while current_row < n_rows:
        start, end = start_end_list[current_row]
        if end != n_frames - 1:
            if list(start_end_list[:, 0]).count(end + 1) == 1:
                next_row = np.where(start_end_list[:, 0] == end + 1)[0][0]

                stitched_face_grid[:, current_row] = np.concatenate((stitched_face_grid[:end+1, current_row], stitched_face_grid[end+1:, next_row]))
                stitched_sparsity[:, current_row] = np.concatenate((stitched_sparsity[:end+1, current_row], stitched_sparsity[end+1:, next_row]))
                stitched_lips_landmarks_grid[:, current_row] = np.concatenate((stitched_lips_landmarks_grid[:end+1, current_row], stitched_lips_landmarks_grid[end+1:, next_row]))

                #deleting row
                stitched_face_grid = np.delete(stitched_face_grid, next_row, axis=1)
                stitched_sparsity = np.delete(stitched_sparsity, next_row, axis=1)
                stitched_lips_landmarks_grid = np.delete(stitched_lips_landmarks_grid, next_row, axis=1)

                start_end_list[current_row] = (start, start_end_list[next_row][1])
                start_end_list = np.delete(start_end_list, next_row, axis=0)

                n_rows -= 1
                current_row -= 1

        current_row += 1

    return stitched_face_grid, stitched_sparsity, stitched_lips_landmarks_grid

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


def build_face_grids(frames_stack):
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
                face_grid[i, j] = Image.fromarray(face_img)
                landmarks_grid[i, j] = landmarks

            else:
                face_grid[i, j] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    return face_grid, sparsity, landmarks_grid
