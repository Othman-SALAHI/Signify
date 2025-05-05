from flask import Flask, render_template, Response, jsonify, send_file
from flask_cors import CORS
from io import BytesIO
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from gtts import gTTS
import time
import threading
import os

app = Flask(__name__, template_folder='server/templates', static_folder='server/static')
CORS(app)

CAPTURE_INTERVAL = 4  # seconds
MIN_CONFIDENCE = 0.7

camera_lock = threading.Lock()
camera_active = False
camera_instance = None

model = load_model('server/py/model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class_names = [chr(i) for i in range(65, 91)] + [' ']
decode = lambda idx: class_names[idx]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

class AppState:
    def __init__(self):
        self.phrase = ""
        self.last_capture_time = 0
        self.capturing = True
        self.lock = threading.Lock()
        self.current_letter = ""
        self.current_confidence = 0
        self.landmarks = []

app_state = AppState()

def generate_frames():
    global camera_instance
    while True:
        with camera_lock:
            if not camera_active:
                time.sleep(0.1)
                continue

            if camera_instance is None or not camera_instance.isOpened():
                camera_instance = cv2.VideoCapture(0)
                if not camera_instance.isOpened():
                    print("Error: Could not open video capture")
                    time.sleep(1)
                    continue

            success, frame = camera_instance.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks and app_state.capturing:
                process_hand_landmarks(results, time.time())

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def process_hand_landmarks(results, current_time):
    for hand_landmarks in results.multi_hand_landmarks:
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            input_data = np.array([landmarks], dtype=np.float32)
            try:
                prediction = model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(prediction)
                confidence = prediction[predicted_idx]
                letter = decode(predicted_idx)

                with app_state.lock:
                    app_state.current_letter = letter
                    app_state.current_confidence = float(confidence)
                    app_state.landmarks = landmarks

                    if confidence > MIN_CONFIDENCE and current_time - app_state.last_capture_time >= CAPTURE_INTERVAL:
                        app_state.phrase += letter
                        app_state.last_capture_time = current_time
            except Exception as e:
                print(f"Prediction error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def demo():
    return render_template('demo.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    with camera_lock:
        camera_active = True
    return jsonify({'success': True})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, camera_instance
    with camera_lock:
        camera_active = False
        if camera_instance and camera_instance.isOpened():
            camera_instance.release()
            camera_instance = None
    return jsonify({'success': True})

@app.route('/get_phrase')
def get_phrase():
    with app_state.lock:
        return jsonify({
            'phrase': app_state.phrase,
            'letter': app_state.current_letter,
            'confidence': app_state.current_confidence,
            'landmarks': app_state.landmarks
        })

@app.route('/clear_phrase', methods=['POST'])
def clear_phrase():
    with app_state.lock:
        app_state.phrase = ""
        app_state.current_letter = ""
        app_state.current_confidence = 0
        app_state.landmarks = []
    return jsonify({'success': True})

@app.route('/toggle_capture', methods=['POST'])
def toggle_capture():
    with app_state.lock:
        app_state.capturing = not app_state.capturing
    return jsonify({
        'success': True,
        'capturing': app_state.capturing
    })

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    with app_state.lock:
        text = app_state.phrase.strip()

    if not text:
        text = "No input detected."

    try:
        tts = gTTS(text=text, lang='en')
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return send_file(audio_bytes, mimetype='audio/mpeg', as_attachment=False, download_name='speech.mp3')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/delete_letter', methods=['POST'])
def delete_letter():
    with app_state.lock:
        app_state.phrase = app_state.phrase[:-1]
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(threaded=True, debug=True)
