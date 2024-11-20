import cv2
import face_recognition
import time

# Global variable to store the registered face encoding
registered_face_encoding = None

def capture_image(delay, prompt):
    """
    Captures an image after a delay and returns the captured frame.
    """
    print(prompt)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    start_time = time.time()
    frame = None
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        elapsed_time = time.time() - start_time

        # Display the live feed with a countdown
        cv2.putText(frame, f"Capturing in {int(delay - elapsed_time)} sec...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Feed", frame)

        if elapsed_time >= delay:
            print("Captured image.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Option to quit early
            print("Exiting...")
            break

    camera.release()
    cv2.destroyAllWindows()
    return frame

def register_face():
    """
    Captures and registers a face for future comparisons.
    """
    global registered_face_encoding

    print("Starting face registration...")
    img = capture_image(10, "Get ready for face registration in 10 seconds...")

    # Convert the image to RGB format
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract the face encoding
    try:
        registered_face_encoding = face_recognition.face_encodings(rgb_img)[0]
        print("Face successfully registered.")
    except IndexError:
        print("Error: No face detected. Please try again.")
        registered_face_encoding = None

def login_with_face():
    """
    Captures an image and compares it with the registered face for authentication.
    """
    global registered_face_encoding

    if registered_face_encoding is None:
        print("No face registered. Please register a face first.")
        return

    print("Starting login with face...")
    img = capture_image(10, "Get ready for login with face in 10 seconds...")

    # Convert the image to RGB format
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract the face encoding
    try:
        login_face_encoding = face_recognition.face_encodings(rgb_img)[0]

        # Compare the login face with the registered face
        result = face_recognition.compare_faces([registered_face_encoding], login_face_encoding)
        if result[0]:
            print("Login successful: Face matched!")
        else:
            print("Login failed: Face did not match.")
    except IndexError:
        print("Error: No face detected. Please try again.")

# Run the program
print("Step 1: Register your face.")
register_face()

print("\nStep 2: Login with your face.")
login_with_face()

print("\nProcess completed. Exiting the program.")
