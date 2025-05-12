import sklearn as sk
import numpy as np
import pandas as pd
import joblib
import warnings
import mediapipe as mp

import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.utils.validation')
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
cap = cv2.VideoCapture(3)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5
)
class SleepDetector:
    def __init__(self, model_path='sleep_detection_model.joblib'):
        self.model = joblib.load(model_path)


    def calculate_ear(self,eye_landmarks, face_landmarks):
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


    def get_facemesh(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks
        return None


    def find_features(self,frame):

        result = self.get_facemesh(frame)
        if result is None:
            return None
        face_landmarks = result[0].landmark
        left_ear = self.calculate_ear(LEFT_EYE , result[0].landmark)
        right_ear = self.calculate_ear(RIGHT_EYE , result[0].landmark)
        avg_ear = (left_ear + right_ear) / 2
        eye_y = (face_landmarks[LEFT_EYE[0]].y + face_landmarks[RIGHT_EYE[0]].y) / 2
        eye_sysmetry = abs(left_ear - right_ear)
        return [avg_ear, left_ear, right_ear, eye_sysmetry, eye_y]



    def processing(self,frame):
        features = self.find_features(frame)
        if features is None:
            return None
        return features


    def main(self):

        while True:
            ret, frame = cap.read()

            features = self.processing(frame)
            if features is None:
                cv2.imshow('frame', frame)
                continue
            prediction = self.model.predict([features])
            active_status = "Active" if prediction[0] == 1 else "Not Active"
            cv2.putText(frame, active_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(prediction)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def detect_sleep(self,frame):
        features = self.processing(frame)
        if features is None:
            return None
        prediction = self.model.predict([features])
        active_status = "Active" if prediction[ 0 ] == 1 else "Not Active"
        return active_status

if __name__ == '__main__':
    detector = SleepDetector()

    detector.main()
