import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Tuple

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def detect_hand_landmarks(image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect hand landmarks in an image using MediaPipe.

    Args:
        image: Input image as numpy array (BGR format)

    Returns:
        Tuple containing:
        - Annotated image with landmarks drawn
        - List of detected hand landmarks
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)

    # Convert back to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    # Extract landmarks as a list of dictionaries
    landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append({"x": landmark.x, "y": landmark.y, "z": landmark.z})
            landmarks_list.append(landmarks)

    return image_bgr, landmarks_list


def extract_hand_features(landmarks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract features from hand landmarks for classification.

    Args:
        landmarks: List of hand landmarks

    Returns:
        Feature vector as numpy array
    """
    if not landmarks:
        return np.array([])

    # Flatten the landmarks into a feature vector
    features = []
    for landmark in landmarks:
        features.extend([landmark["x"], landmark["y"], landmark["z"]])

    # Calculate additional features (distances between key points)
    # Example: distance between thumb tip and index finger tip
    thumb_tip = np.array([landmarks[4]["x"], landmarks[4]["y"], landmarks[4]["z"]])
    index_tip = np.array([landmarks[8]["x"], landmarks[8]["y"], landmarks[8]["z"]])
    distance = np.linalg.norm(thumb_tip - index_tip)
    features.append(distance)

    # Add more custom features as needed

    return np.array(features)


def process_video_for_dynamic_gestures(
    video_path: str, max_frames: int = 30
) -> List[np.ndarray]:
    """
    Process a video file to extract hand landmarks for dynamic gesture recognition.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to process

    Returns:
        List of feature vectors for each frame
    """
    cap = cv2.VideoCapture(video_path)
    frame_features = []

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            success, image = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image
            results = hands.process(image_rgb)

            # Extract landmarks
            if results.multi_hand_landmarks:
                landmarks = []
                for landmark in results.multi_hand_landmarks[0].landmark:
                    landmarks.append(
                        {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                    )

                # Extract features from landmarks
                features = extract_hand_features(landmarks)
                frame_features.append(features)
            else:
                # If no hand detected, add a zero vector
                frame_features.append(np.zeros(63))  # Assuming 21 landmarks with x,y,z

            frame_count += 1

    cap.release()
    return frame_features
