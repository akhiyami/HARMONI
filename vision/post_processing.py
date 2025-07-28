"""
Post-processing functions for detected faces grids.
"""

#--------------------------------------- Imports ---------------------------------------#

import numpy as np

from vision.config.settings import FILTER_THRESHOLD

#--------------------------------------- Functions ---------------------------------------#

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

def compute_speaking_probability(landmark_sequences, threshold=1.5):
    """
    Estimate speaking probability from mouth landmark motion.
    
    Parameters:
        landmark_sequences: np.ndarray of shape [T, M, 2]
            - T = number of frames
            - M = number of mouth landmarks
            - Each entry is [x, y] coordinates
        threshold: float
            - Sensitivity threshold for motion magnitude to infer speaking
    
    Returns:
        probs: list of floats, speaking probability per frame (T-1 entries)
    """
    landmark_sequences = np.array(landmark_sequences)  # shape [T, M, 2]
    T = landmark_sequences.shape[0]

    motion_magnitudes = []
    for t in range(1, T):
        #check landmarks values are not empty
        # Compute displacement of each landmark between frames
        diffs = landmark_sequences[t] - landmark_sequences[t - 1]  # shape [M, 2]
        distances = np.linalg.norm(diffs, axis=1)  # shape [M]
        mean_motion = np.mean(distances)
        motion_magnitudes.append(mean_motion)

    # Normalize and convert to probability (sigmoid-style)
    motion_magnitudes = np.array(motion_magnitudes)
    probs = 1 / (1 + np.exp(- (motion_magnitudes - threshold)))

    return probs.tolist()
