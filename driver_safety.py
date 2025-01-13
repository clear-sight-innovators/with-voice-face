import cv2
import mediapipe as mp
import time
import numpy as np
from scipy.spatial import distance
import pygame

# Constants
YAWN_THRESHOLD = 0.6
EYE_ASPECT_RATIO_THRESHOLD = 0.22  # Adjust threshold for eye closure detection
EYE_CLOSED_FRAMES_THRESHOLD = 10  # Consecutive frames with eyes closed to be considered drowsy
ALERT_SOUND_PATH = 'alert_sound.mp3'
LOOK_AWAY_FRAMES_THRESHOLD = 20  # Number of consecutive frames for looking away detection

# Eye Aspect Ratio function to detect blinking and drowsiness
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Yawn detection function based on mouth aspect ratio
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])  # Upper to bottom middle
    B = distance.euclidean(mouth[2], mouth[9])  # Upper left to bottom right middle
    C = distance.euclidean(mouth[0], mouth[6])  # Left corner to right corner
    mar = (A + B) / (2.0 * C)
    return mar

# Initialize Pygame for sound alerts
pygame.mixer.init()

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)
start_time = time.time()

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    frame_count = 0  # Counter for consecutive frames with eyes closed
    look_away_count = 0  # Counter for consecutive frames looking away

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to RGB (MediaPipe works with RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Convert the frame back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get face landmarks as a numpy array
                landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark])

                # Extract eye landmarks (left eye and right eye)
                left_eye = landmarks[362:374]
                right_eye = landmarks[133:145]
                left_eye_ear = eye_aspect_ratio(left_eye)
                right_eye_ear = eye_aspect_ratio(right_eye)
                ear = (left_eye_ear + right_eye_ear) / 2.0

                # Check for eye blinking and drowsiness
                if ear < EYE_ASPECT_RATIO_THRESHOLD:
                    frame_count += 1
                    if frame_count >= EYE_CLOSED_FRAMES_THRESHOLD:
                        cv2.putText(frame_bgr, "Drowsy: Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        pygame.mixer.music.load('wake_up.mp3')
                        pygame.mixer.music.play()
                else:
                    frame_count = 0  # Reset counter when eyes are open

                # Extract mouth landmarks for yawn detection
                mouth = landmarks[78:88]
                mar = mouth_aspect_ratio(mouth)
                if mar > YAWN_THRESHOLD:
                    cv2.putText(frame_bgr, "Yawning: Take a Break", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    pygame.mixer.music.load('yawn_alert.mp3')
                    pygame.mixer.music.play()

                # Check for distraction (looking elsewhere)
                left_eye_pos = landmarks[362]
                right_eye_pos = landmarks[133]
                nose_pos = landmarks[168]

                # Detect looking down (eyes too low, head tilted down)
                if left_eye_pos[1] > 0.6 or right_eye_pos[1] > 0.6:
                    cv2.putText(frame_bgr, "Distraction: Looking Down", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    pygame.mixer.music.load('look_down_alert.mp3')
                    pygame.mixer.music.play()
                    look_away_count = 0  # Reset count when looking down

                # Detect looking away (nose position too far horizontally)
                elif nose_pos[1] < 0.15:  
                    look_away_count += 1  # Increment the counter when looking away
                    if look_away_count >= LOOK_AWAY_FRAMES_THRESHOLD:
                        cv2.putText(frame_bgr, "Distraction: Looking Away", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        pygame.mixer.music.load('look_away_alert.mp3')
                        pygame.mixer.music.play()
                else:
                    look_away_count = 0  # Reset count if nose is back in a normal position

                # Draw face landmarks
                mp_drawing.draw_landmarks(frame_bgr, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # Display frame
        cv2.imshow('Driver Distraction Detection', frame_bgr)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
