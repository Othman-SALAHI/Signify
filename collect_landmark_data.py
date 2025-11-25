import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Set up webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create directories and files to store data
if not os.path.exists('data'):
    os.makedirs('data')

# Labels (adjust this based on the sign language you want to use)
class_names = [
    'space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]


# Create CSV file to store data
csv_file = open('data/landmarksss.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write header: x0,y0,z0,...,x20,y20,z20,label
header = []
for i in range(21):  # 21 hand landmarks
    header.extend([f'x{i}', f'y{i}', f'z{i}'])
header.append('label')
csv_writer.writerow(header)

print("[INFO] Collecting data. Press 'q' to stop.")

# Timer setup
start_time = time.time()
current_class = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract 63 landmark values
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                cv2.putText(frame, f"Collecting {class_names[current_class]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow("Landmark Collection", frame)

                # Collect data every 1 second
                if time.time() - start_time >= 0.2:
                    # Save landmarks and label at the end
                    csv_writer.writerow(landmarks + [class_names[current_class]])
                    print(f"[INFO] Collected data for {class_names[current_class]}")

                    # Reset timer
                    start_time = time.time()

                    print("[INFO] Press 'n' to move to the next letter or 'q' to quit.")
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        csv_file.close()
                        cv2.destroyAllWindows()
                        exit()
                    elif key == ord('n'):
                        current_class += 1
                        if current_class >= len(class_names):
                            print("[INFO] Data collection completed.")
                            cap.release()
                            csv_file.close()
                            cv2.destroyAllWindows()
                            exit()

    # Show the camera feed
    cv2.imshow("Hand Sign Recognition", frame)

    time.sleep(0.01)  # Small delay for smoother performance

# Clean up
cap.release()
csv_file.close()
cv2.destroyAllWindows()
