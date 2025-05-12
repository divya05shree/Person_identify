import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# State variables
feature_history = deque(maxlen=10)  # Store recent features for smoothing
sleep_state_history = deque(maxlen=30)  # Store sleep state predictions
sleep_start_time = None
is_sleeping = False
collecting_data = False
training_data = []
training_labels = []
model = None
model_file = "sleep_detection_model.joblib"

# Parameters
COLLECTION_INTERVAL = 2.0  # Seconds between data collection
last_collection_time = 0


def calculate_ear(eye_landmarks, face_landmarks):
    """Calculate Eye Aspect Ratio for one eye"""
    points = [face_landmarks[i] for i in eye_landmarks]

    # Horizontal distances
    horizontal_1 = np.sqrt((points[0].x - points[3].x) ** 2 + (points[0].y - points[3].y) ** 2)
    horizontal_2 = np.sqrt((points[1].x - points[4].x) ** 2 + (points[1].y - points[4].y) ** 2)
    horizontal_3 = np.sqrt((points[2].x - points[5].x) ** 2 + (points[2].y - points[5].y) ** 2)

    # Vertical distances
    vertical_1 = np.sqrt((points[1].x - points[5].x) ** 2 + (points[1].y - points[5].y) ** 2)
    vertical_2 = np.sqrt((points[2].x - points[4].x) ** 2 + (points[2].y - points[4].y) ** 2)

    # Calculate EAR
    horizontal_mean = (horizontal_1 + horizontal_2 + horizontal_3) / 3
    vertical_mean = (vertical_1 + vertical_2) / 2

    ear = vertical_mean / horizontal_mean if horizontal_mean > 0 else 0.3
    return ear


def calculate_head_pose(landmarks, img_h, img_w):
    """Calculate head pose angles from facial landmarks"""
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Convert landmarks to image points
    image_points = []
    for idx in POSE_LANDMARKS:
        x = int(landmarks[idx].x * img_w)
        y = int(landmarks[idx].y * img_h)
        image_points.append((x, y))

    image_points = np.array(image_points, dtype=np.float64)

    # Camera matrix estimation
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Find pose
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        # Convert rotation vector to euler angles
        rmat = cv2.Rodrigues(rotation_vector)[0]
        proj_mat = np.hstack((rmat, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_mat)[6]

        pitch = euler_angles[0, 0]  # x-axis rotation
        yaw = euler_angles[1, 0]  # y-axis rotation
        roll = euler_angles[2, 0]  # z-axis rotation

        return pitch, yaw, roll
    except:
        return 0, 0, 0


def extract_features(face_landmarks, h, w):
    """Extract features for sleep detection"""
    # 1. EAR features
    left_ear = calculate_ear(LEFT_EYE, face_landmarks)
    right_ear = calculate_ear(RIGHT_EYE, face_landmarks)
    avg_ear = (left_ear + right_ear) / 2.0

    # 2. Head pose
    pitch, yaw, roll = calculate_head_pose(face_landmarks, h, w)

    # 3. Eye position relative to face (vertical proportion)
    eye_y = (face_landmarks[LEFT_EYE[0]].y + face_landmarks[RIGHT_EYE[0]].y) / 2
    nose_y = face_landmarks[1].y
    chin_y = face_landmarks[152].y
    face_height = chin_y - face_landmarks[10].y  # Top of forehead to chin
    eye_position_ratio = (eye_y - nose_y) / face_height if face_height > 0 else 0

    # 4. Eye openness symmetry (left vs right eye)
    eye_symmetry = abs(left_ear - right_ear)

    # 5. Mouth openness (can indicate yawning)
    mouth_top = face_landmarks[13].y
    mouth_bottom = face_landmarks[14].y
    mouth_openness = (mouth_bottom - mouth_top) / face_height if face_height > 0 else 0

    # Return feature vector
    return [avg_ear, left_ear, right_ear, eye_symmetry,
            pitch, yaw, roll, eye_position_ratio, mouth_openness]


def train_model():
    """Train a machine learning model on collected data"""
    global model

    if len(training_data) < 10 or len(set(training_labels)) < 2:
        return False  # Not enough data or not enough classes

    X = np.array(training_data)
    y = np.array(training_labels)

    # Create and train the model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_file)

    return True


def load_model():
    """Load the trained model if it exists"""
    global model
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        return True
    return False


# Initialize video capture
cap = cv2.VideoCapture(0)

# Try to load existing model
model_loaded = load_model()

# Instructions to display when starting
print("\nSleep Detection System")
print("----------------------")
print("Controls:")
print("  'a' - Start/stop collecting awake data")
print("  's' - Start/stop collecting sleep data")
print("  't' - Train model with collected data")
print("  'c' - Clear all training data")
print("  'l' - Load existing model")
print("  'r' - Reset sleep detection state")
print("  'q' - Quit the application")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Process frame with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Get frame dimensions
    h, w, c = frame.shape

    # Prepare UI elements
    status_area = np.zeros((200, w, 3), dtype=np.uint8)

    # Training status indicator
    collection_status = "COLLECTING DATA" if collecting_data else "MONITORING"
    collection_color = (0, 165, 255) if collecting_data else (255, 255, 255)
    cv2.putText(status_area, f"Mode: {collection_status}", (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, collection_color, 2)

    cv2.putText(status_area, f"Data samples: {len(training_data)}", (w - 250, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Extract features
        features = extract_features(face_landmarks, h, w)

        # Draw eye landmarks
        for idx in LEFT_EYE + RIGHT_EYE:
            x, y = int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Collect training data if in collection mode
        if collecting_data and current_time - last_collection_time > COLLECTION_INTERVAL:
            # Add current features and label to training data
            training_data.append(features)
            # Use '1' for sleep state when collecting with 's' key pressed, otherwise '0'
            if collecting_data == "sleep":
                training_labels.append(1)
            else:
                training_labels.append(0)

            last_collection_time = current_time

            # Update status area to show collection
            cv2.putText(status_area, f"Sample collected!", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Use model for prediction if available
        if model is not None:
            # Add features to history for smoothing
            feature_history.append(features)

            # Make prediction using averaged features
            if len(feature_history) > 0:
                avg_features = np.mean(np.array(feature_history), axis=0).reshape(1, -1)
                prediction = model.predict(avg_features)[0]
                prediction_prob = model.predict_proba(avg_features)[0]

                # Add to sleep state history for temporal smoothing
                sleep_state_history.append(prediction)

                # Smoothed prediction - majority vote
                sleep_votes = sum(sleep_state_history)
                total_votes = len(sleep_state_history)
                sleep_probability = sleep_votes / total_votes if total_votes > 0 else 0

                # Determine sleep state with hysteresis
                previous_sleep_state = is_sleeping
                if not is_sleeping and sleep_probability > 0.7:  # Need strong evidence to enter sleep state
                    is_sleeping = True
                    sleep_start_time = current_time
                elif is_sleeping and sleep_probability < 0.3:  # Need strong evidence to exit sleep state
                    is_sleeping = False
                    sleep_start_time = None

                # Calculate sleep duration
                sleep_duration = (current_time - sleep_start_time) if sleep_start_time else 0

                # Display prediction information
                awake_prob = 1 - prediction_prob[1] if len(prediction_prob) > 1 else 1
                sleep_prob = prediction_prob[1] if len(prediction_prob) > 1 else 0

                cv2.putText(status_area, f"Awake prob: {awake_prob:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(status_area, f"Sleep prob: {sleep_prob:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display sleep state
                state_text = "SLEEPING" if is_sleeping else "AWAKE"
                state_color = (0, 0, 255) if is_sleeping else (0, 255, 0)
                cv2.putText(status_area, f"State: {state_text}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

                # Display average EAR (most important feature)
                ear = features[0]  # First feature is average EAR
                cv2.putText(status_area, f"EAR: {ear:.3f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display warning if sleeping
                if is_sleeping:
                    cv2.putText(frame, "SLEEP DETECTED!", (w // 2 - 150, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.putText(frame, f"Duration: {sleep_duration:.1f}s", (w // 2 - 120, h // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No model available
            cv2.putText(status_area, "No model available. Press 't' to train.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Display basic features
            ear = (calculate_ear(LEFT_EYE, face_landmarks) +
                   calculate_ear(RIGHT_EYE, face_landmarks)) / 2
            pitch, _, _ = calculate_head_pose(face_landmarks, h, w)

            cv2.putText(status_area, f"EAR: {ear:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(status_area, f"Pitch: {pitch:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # No face detected
        cv2.putText(status_area, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Combine frame and status area
    combined_display = np.vstack([frame, status_area])
    cv2.imshow("ML Sleep Detection", combined_display)

    # Key handlers
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a'):
        # Start/stop collecting awake data
        if collecting_data == "awake":
            collecting_data = False
            print("Stopped collecting awake data")
        else:
            collecting_data = "awake"
            print("Started collecting awake data")
    elif key == ord('s'):
        # Start/stop collecting sleep data
        if collecting_data == "sleep":
            collecting_data = False
            print("Stopped collecting sleep data")
        else:
            collecting_data = "sleep"
            print("Started collecting sleep data")
    elif key == ord('t'):
        # Train model with collected data
        print("Training model...")
        if train_model():
            print("Model trained successfully!")
        else:
            print("Not enough data to train model. Collect more samples.")
    elif key == ord('c'):
        # Clear training data
        training_data = []
        training_labels = []
        print("Training data cleared")
    elif key == ord('l'):
        # Load existing model
        if load_model():
            print("Model loaded successfully")
        else:
            print("No model file found")
    elif key == ord('r'):
        # Reset sleep detection state
        is_sleeping = False
        sleep_start_time = None
        sleep_state_history.clear()
        feature_history.clear()
        print("Sleep detection state reset")

# Clean up
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

if __name__ == "__main__":
    # If model file exists, try to load it
    if os.path.exists(model_file):
        print(f"Found existing model: {model_file}")
        load_model()
    else:
        print("No existing model found. Collect data and train a new model.")