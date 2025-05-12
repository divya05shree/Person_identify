import mediapipe as mp
import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
cap = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

def get_facemesh(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    return results.multi_face_landmarks


def find_features(frame):

    result = get_facemesh(frame)
    print(result)
    face_landmarks = result[0].landmark
    left_ear = calculate_ear(LEFT_EYE , result[0].landmark)
    right_ear = calculate_ear(RIGHT_EYE , result[0].landmark)
    avg_ear = (left_ear + right_ear) / 2
    eye_y = (face_landmarks[LEFT_EYE[0]].y + face_landmarks[RIGHT_EYE[0]].y) / 2
    eye_sysmetry = abs(left_ear - right_ear)
    return [avg_ear, left_ear, right_ear, eye_sysmetry, eye_y]


def save_features_csv(features):
    with open('features.csv', 'a') as f:
        for feature in features:
            f.write(','.join(map(str, feature)) + ",0"+'\n')


def main():
    count = 0
    features =[]
    while count < 100:
        count += 1
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        features.append(find_features(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    save_features_csv(features)


if __name__ == "__main__":
    main()
