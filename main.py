import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# --- Load trained CNN model ---
model = load_model("isl_cnn_model.keras")

# --- Constants ---
IMG_SIZE = (64, 64)
CLASS_LABELS = [
    chr(i)
    for i in range(ord("A"), ord("Z") + 1)
    if i not in (ord("J"), ord("X"), ord("Z"))
]  # A–Z excluding J, X, Z

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

# --- Start Webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    result = hands.process(rgb)

    prediction = "..."

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add padding
            pad = 20
            x_min = max(x_min - pad, 0)
            y_min = max(y_min - pad, 0)
            x_max = min(x_max + pad, w)
            y_max = min(y_max + pad, h)

            # Crop and preprocess
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, IMG_SIZE)
            normalized = resized.astype("float32") / 255.0
            input_tensor = normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

            # Predict with CNN
            probs = model.predict(input_tensor)
            predicted_index = np.argmax(probs)
            prediction = CLASS_LABELS[predicted_index]

    # --- Overlay Prediction ---
    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    cv2.imshow("ISL Alphabet Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
