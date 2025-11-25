import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow
import pickle

# === Load your trained model ===
model = load_model("model.h5")  # Make sure this file exists in the same folder

# === Load class names (e.g., ['A', 'B', 'C', ...]) ===
dataset = tensorflow.keras.preprocessing.image_dataset_from_directory(
    'DATASET',
    shuffle = True, image_size = (64,64), batch_size = 25)
class_names = dataset.class_names

# === Decode function from class index to label ===
decode = lambda idx: class_names[idx]

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time hand sign recognition... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 hand landmarks → 63 values
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                # Predict the class
                prediction = model.predict(np.array([landmarks]))[0]
                predicted_idx = np.argmax(prediction)
                predicted_label = decode(predicted_idx)

                # Display the prediction
                cv2.putText(frame, f'Prediction: {predicted_label}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Hand Sign Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
