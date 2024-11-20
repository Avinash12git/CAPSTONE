import face_recognition
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import cv2
import mediapipe as mp
import time
import math
import base64
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


users = {}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Calculate angle between joints
def calculate_angle(A, B, C):
    AB = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]
    magnitude_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
    magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    angle = math.acos(dot_product / (magnitude_AB * magnitude_BC))
    return math.degrees(angle)

# Detect finger states
def detect_finger_state(landmarks, handedness):
    finger_bases = [2, 5, 9, 13, 17]
    finger_tips = [4, 8, 12, 16, 20]
    finger_intermediates = [3, 6, 10, 14, 18]
    finger_states = []

    if handedness == 'Right':
        thumb_is_straight = landmarks[4].x < landmarks[2].x
    else:
        thumb_is_straight = landmarks[4].x > landmarks[2].x
    finger_states.append(2 if thumb_is_straight else 1)

    for base, intermediate, tip in zip(finger_bases[1:], finger_intermediates[1:], finger_tips[1:]):
        angle = calculate_angle(landmarks[base], landmarks[intermediate], landmarks[tip])
        if angle > 160:
            finger_states.append(2)
        elif angle > 90:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states


# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('name')
        password = request.form.get('password')
        passkey = request.form.get('passkey').lower()  # Convert passkey to lowercase
        
        # Store user data securely
        users[username] = {'password': password, 'passkey': passkey}
        
        # Save username in session for subsequent steps
        session['username'] = username
        
        # Redirect to gesture registration
        return redirect(url_for('gesture_register'))
    
    return render_template('register.html')

@app.route('/gesture_register', methods=['GET', 'POST'])
def gesture_register():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        image_data = request.json.get('image')
        if image_data:
            img_bytes = base64.b64decode(image_data.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        gesture = detect_finger_state(hand_landmarks.landmark, handedness.classification[0].label)
                        users[username]['gesture'] = gesture
                        return jsonify(success=True)
        return jsonify(success=False)
    return render_template('gesture_register.html')


face_encodings_db = {}

# Directory to save face encodings
os.makedirs('face_data', exist_ok=True)

@app.route('/face_register_page')
def face_register_page():
    return render_template('face_register.html')

@app.route('/face_register', methods=['POST'])
def face_register():
    if request.is_json:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify(success=False, message="No image data received")

        # Convert the image from base64 to OpenCV format
        img_bytes = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract face encoding
        face_encodings = face_recognition.face_encodings(rgb_frame)
        if face_encodings:
            username = session.get('username')
            if username:
                # Save the face encoding as a .npy file
                np.save(f'face_data/{username}.npy', face_encodings[0])
                # Redirect to index page upon successful registration
                return jsonify(success=True, message="Face registered successfully", redirect=url_for('index'))
        return jsonify(success=False, message="No face detected")
    return jsonify(success=False, message="Invalid request")

# Registration Success
@app.route('/register_success')
def register_success():
    gesture_data = session.get('gesture_data', [])
    face_data = session.get('face_data', "Not Registered")
    voice_data = session.get('voice_data', "Not Registered")
    return render_template('register_success.html', gesture_data=gesture_data, face_data=face_data, voice_data=voice_data)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Register Task
@app.route('/register_task')
def register_task():
    return render_template('register_task.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['name']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['username'] = username
            return redirect(url_for('login_task'))
        else:
            return render_template('login_fail.html')
    return render_template('login.html')

@app.route('/login_task', methods=['GET'])
def login_task():
    return render_template('login_task.html')

@app.route('/combined_login', methods=['GET', 'POST'])
def combined_login():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))

    if request.method == 'POST':
        data = request.json
        image_data_gesture = data.get('gesture_image')
        image_data_face = data.get('face_image')

        # Gesture Authentication
        gesture_authenticated = False
        if image_data_gesture:
            img_bytes = base64.b64decode(image_data_gesture.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        gesture = detect_finger_state(hand_landmarks.landmark, handedness.classification[0].label)
                        if gesture == users[username]['gesture']:
                            gesture_authenticated = True
                            break

        # Face Authentication
        face_authenticated = False
        if image_data_face:
            img_bytes = base64.b64decode(image_data_face.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_encodings = face_recognition.face_encodings(rgb_frame)
            if face_encodings:
                registered_encoding_path = f'face_data/{username}.npy'
                if os.path.exists(registered_encoding_path):
                    registered_encoding = np.load(registered_encoding_path)
                    matches = face_recognition.compare_faces([registered_encoding], face_encodings[0])
                    face_authenticated = matches[0]

        # Final Decision
        if gesture_authenticated and face_authenticated:
            return jsonify(success=True, redirect=url_for('login_task'))  # Redirects to the welcome page
        else:
            return jsonify(success=False, redirect=url_for('login_fail'))  # Redirects to login failure page

    return render_template('combined_login.html')

import speech_recognition as sr

# Combined face and voice authentication route
@app.route('/combined1_login', methods=['GET', 'POST'])
def combined1_login():
    if request.method == 'POST':
        # Step 1: Authenticate face
        image_data = request.json.get('image')
        if image_data:
            # Convert image to OpenCV format
            img_bytes = base64.b64decode(image_data.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if face_encodings:
                username = session.get('username')
                if username:
                    registered_encoding_path = f'face_data/{username}.npy'
                    if os.path.exists(registered_encoding_path):
                        registered_encoding = np.load(registered_encoding_path)
                        face_match = face_recognition.compare_faces([registered_encoding], face_encodings[0])
                        if not face_match[0]:
                            return jsonify(success=False, message="Face authentication failed.")
                    else:
                        return jsonify(success=False, message="No registered face found.")
            else:
                return jsonify(success=False, message="No face detected.")

        # Step 2: Authenticate voice
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                # Capture audio and convert to text
                audio_data = recognizer.listen(source, timeout=5)
                spoken_passkey = recognizer.recognize_google(audio_data).strip().lower()
                username = session.get('username')
                if username and users[username]['passkey'] == spoken_passkey:
                    # If both face and voice match, authenticate successfully
                    return jsonify(success=True, redirect=url_for('welcome'))
                else:
                    return jsonify(success=False, message="Voice authentication failed.")
            except sr.UnknownValueError:
                return jsonify(success=False, message="Could not understand the audio. Please try again.")
            except sr.RequestError as e:
                return jsonify(success=False, message=f"Voice recognition service error: {e}")
    return render_template('combined1_login.html')





@app.route('/welcome')
def welcome():
    return render_template('welcome.html')


@app.route('/login_fail')
def login_fail():
    return render_template('login_fail.html', message="Invalid gesture or face authentication. Please try again.")


if __name__ == '__main__':
    app.run(debug=True)
