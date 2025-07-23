import dlib

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
        

import numpy as np

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