from flask import Flask, Response, render_template
import cv2
import numpy as np
from fer import FER  # Emotion recognition
import mediapipe as mp

# Initialize Flask app
app = Flask(__name__)

# Initialize the FER detector
detector = FER()

# Load the face cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Face Mesh for tracking facial features
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True)

# Store previous positions for motion tracking
previous_positions = []

def detect_faces_and_emotions(frame):
    """Detect faces, classify emotions, and track motion with dots and lines."""
    global previous_positions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    current_positions = []
    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]
        
        # Use FER library to detect the emotion
        emotion, score = detector.top_emotion(face)
        
        # Draw a rectangle around the face and display emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if score is not None:
            cv2.putText(frame, f"{emotion} ({score*100:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Calculate the center of the face
        center_x, center_y = x + w // 2, y + h // 2
        current_positions.append((center_x, center_y))
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Draw tracking point
    
    # Draw lines to show motion paths
    for i in range(1, len(previous_positions)):
        if previous_positions[i - 1] is not None and previous_positions[i] is not None:
            cv2.line(frame, previous_positions[i - 1], previous_positions[i], (0, 255, 255), 2)
    
    # Update previous positions
    previous_positions = current_positions
    return frame, len(faces)

def track_face_mesh(frame):
    """Track a few key facial landmarks and connect them with lines."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract specific key landmark indices
            key_points = [1, 33, 61, 199, 291, 263]  # Nose, eyes, mouth corners, jaw
            landmark_coords = []

            for idx in key_points:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                landmark_coords.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw fewer dots

            # Draw lines connecting key points
            for i in range(len(landmark_coords) - 1):
                cv2.line(frame, landmark_coords[i], landmark_coords[i + 1], (0, 255, 255), 2)
    
    return frame

def generate_frames():
    """Generate video frames from the webcam."""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces, emotions, and track motion
        frame, num_people = detect_faces_and_emotions(frame)

        # Track facial landmarks
        frame = track_face_mesh(frame)
        
        # Add text for the number of people detected
        cv2.putText(frame, f"People Count: {num_people}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provide the video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
