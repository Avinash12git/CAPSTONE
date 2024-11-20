import face_recognition
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import cv2
import mediapipe as mp
import time
import math
import base64
import numpy as np
import os

import speech_recognition as sr

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


users = {}
face_encodings_db = {}

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

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

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

# Registration Success
@app.route('/register_success')
def register_success():
    gesture_data = session.get('gesture_data', [])
    face_data = session.get('face_data', "Not Registered")
    return render_template('register_success.html', gesture_data=gesture_data, face_data=face_data)

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


# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['name']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['username'] = username
            return redirect(url_for('gesture_login'))
        else:
            return render_template('login_fail.html')
    return render_template('login.html')

@app.route('/login_task', methods=['GET'])
def login_task():
    return render_template('login_task.html')

# Gesture Login
@app.route('/gesture_login', methods=['GET', 'POST'])
def gesture_login():
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
                        if gesture == users[username]['gesture']:
                            return redirect(url_for('login_task'))
        return render_template('login_fail.html')
    return render_template('gesture_login.html')

@app.route('/face_login', methods=['GET', 'POST'])
def face_login():
    if request.method == 'POST':
        image_data = request.json.get('image')

        if image_data:
            # Convert the image from base64 to OpenCV format
            img_bytes = base64.b64decode(image_data.split(',')[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract face encoding from the captured image
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if face_encodings:
                username = session.get('username')
                if username:
                    try:
                        registered_encoding_path = f'face_data/{username}.npy'
                        if os.path.exists(registered_encoding_path):
                            registered_encoding = np.load(registered_encoding_path)
                            matches = face_recognition.compare_faces([registered_encoding], face_encodings[0])
                            if matches[0]:
                                # If the face matches, redirect to the welcome page
                                return jsonify(success=True, redirect=url_for('login_task'))
                            else:
                                # If the face does not match, redirect to the login_fail.html page
                                return jsonify(success=False, redirect=url_for('login_fail'))
                        else:
                            # If no registered face is found, redirect to the login_fail.html page
                            return jsonify(success=False, redirect=url_for('login_fail'))
                    except Exception as e:
                        # If an error occurs, redirect to the login_fail.html page
                        return jsonify(success=False, redirect=url_for('login_fail'))
            # If no face is detected, redirect to the login_fail.html page
            return jsonify(success=False, redirect=url_for('login_fail'))
    return render_template('face_login.html')

@app.route('/voice_login', methods=['GET', 'POST'])
def voice_login():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))

    if request.method == 'POST':
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                # Capture the audio
                audio_data = recognizer.listen(source, timeout=5)
                # Convert audio to text
                spoken_passkey = recognizer.recognize_google(audio_data).strip()
                print(f"Detected passkey: {spoken_passkey}")

                # Compare with stored passkey
                if users[username]['passkey'] == spoken_passkey:
                    # Redirect to welcome page upon successful authentication
                    return jsonify(success=True, redirect=url_for('welcome'))
                else:
                    return jsonify(success=False, message="Passkey does not match. Please try again.")
            except sr.UnknownValueError:
                return jsonify(success=False, message="Could not understand the audio. Please try again.")
            except sr.RequestError as e:
                return jsonify(success=False, message=f"Voice recognition service error: {e}")
    return render_template('voice_login.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

