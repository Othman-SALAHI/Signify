import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# === Load the trained model ===
try:
    model = load_model('model.h5')
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load the model: {e}")
    exit()

# === Class labels A-Z + space ===
class_names = [chr(i) for i in range(65, 91)] + [' ']  # A-Z + space
decode = lambda idx: class_names[idx]

# === Initialize MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting real-time hand sign recognition...")
print("    - A letter will be captured every 4 seconds.")
print("    - Press 'O' to stop recording and display the full phrase.")
print("    - Press 'D' to delete the last letter from the phrase.")
print("    - Press 'Q' to quit.")

# === Variables for tracking ===
prev_time = 0
last_capture_time = 0
capture_interval = 4  # seconds
capturing = True
phrase = ""  # This will store the phrase with spaces
display_phrase = ""  # This will display the phrase with spaces
display_label = ""  # Initialize display_label to avoid NameError

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_time = time.time()

    if results.multi_hand_landmarks and capturing:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                input_data = np.array([landmarks], dtype=np.float32)
                prediction = model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(prediction)
                confidence = prediction[predicted_idx]

                if confidence > 0.7:
                    predicted_label = decode(predicted_idx)
                    display_label = "Space" if predicted_label == ' ' else predicted_label

                    # Capture letter every 4 seconds
                    if current_time - last_capture_time >= capture_interval:
                        phrase += predicted_label
                        display_phrase = phrase  # Show spaces properly in the display
                        last_capture_time = current_time
                        print(f"[INFO] Captured letter: '{display_label}'")
                else:
                    display_label = "Uncertain"
                    cv2.putText(frame, display_label, (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    # --- Display Design: Show the phrase with a white background at the bottom ---
    phrase_height = 40
    cv2.rectangle(frame, (0, frame.shape[0] - phrase_height), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
    if display_phrase:
        cv2.putText(frame, f'Phrase: {display_phrase}', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # === Show FPS ===
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # --- Show predicted letter or uncertain at the top ---
    if display_label:
        cv2.putText(frame, f'Letter: {display_label}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Sign Recognition", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o'):  # Stop recording
        capturing = False
        print(f"\n[INFO] Recording stopped.\n[INFO] Final Phrase: {phrase}")
    elif key == ord('s'):  # Stop recording
        capturing = True
        print(f"\n[INFO] Recording started.\n[INFO] Final Phrase: {phrase}")
    elif key == ord('d'):  # Delete last letter
        phrase = phrase[:-1]
        display_phrase = phrase  # Update display phrase
        print(f"[INFO] Deleted last letter. Current Phrase: {display_phrase}")

cap.release()
cv2.destroyAllWindows()
