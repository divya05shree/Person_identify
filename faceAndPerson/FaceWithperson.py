import mediapipe as mp
from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
import pickle
from facenet_pytorch import InceptionResnetV1
class FaceWithPerson:
    def __init__(self):
        # Initialize face detection with MediaPipe
        mp_face_detection = mp.solutions.face_detection
        self.face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        # Initialize YOLO for person detection with GPU acceleration if available
        self.person_detector = YOLO("yolov8n.pt")

        # Only detect people (class 0 in COCO dataset)
        self.person_class_id = 0

        # Track FPS
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0
        self.persons_tracking = []
        # Frame processing settings
        self.frame_skip = 0  # Process every frame by default
        self.frame_count = 0
        self.processing_resolution = (640, 480)  # Reduce resolution for faster processing
        self.trakers =[]
    def detect_person(self, frame):
        """Detect persons in frame using YOLO"""
        # Skip frames to improve performance if needed
        self.frame_count += 1
        if self.frame_skip > 0 and self.frame_count % (self.frame_skip + 1) != 0:
            return [ ]

        persons = [ ]

        # Resize for faster processing
        resized_frame = cv2.resize(frame, self.processing_resolution)

        # Use confidence threshold and NMS parameters for better performance
        results = self.person_detector.predict(
            resized_frame,
            verbose=False,
            conf=0.5,  # Higher confidence threshold for faster processing
            classes=[ self.person_class_id ]  # Only detect persons
        )[ 0 ]

        # Calculate scaling factors
        height_ratio = frame.shape[ 0 ] / self.processing_resolution[ 1 ]
        width_ratio = frame.shape[ 1 ] / self.processing_resolution[ 0 ]
        person_tackering = []
        # Process results
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box

            # Only process if it's a person (class 0)
            if int(cls) == self.person_class_id:
                # Scale back to original dimensions
                x1 = int(x1 * width_ratio)
                y1 = int(y1 * height_ratio)
                x2 = int(x2 * width_ratio)
                y2 = int(y2 * height_ratio)

                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[ 1 ], x2)
                y2 = min(frame.shape[ 0 ], y2)

                width, height = x2 - x1, y2 - y1

                # Only process if bounding box has reasonable size
                if width > 0 and height > 0:
                    bbox = [ x1, y1, width, height ]
                    cropped_person = frame[ y1:y2, x1:x2 ]
                    persons.append((bbox, cropped_person))
                    person_tackering.append([bbox , conf , 'person'])
        self.persons_tracking = person_tackering
        return persons

    def detect_face(self, person_img):
        """Detect face in a person image"""
        # Skip if image is too small
        if person_img.shape[ 0 ] < 20 or person_img.shape[ 1 ] < 20:
            return None

        # Avoid redundant conversion if the image is already RGB
        if len(person_img.shape) == 3 and person_img.shape[ 2 ] == 3:
            frame_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe
            results = self.face_detector.process(frame_rgb)

            if results.detections:
                # Get the first detected face
                detection = results.detections[ 0 ]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = person_img.shape

                # Convert relative coordinates to absolute
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Ensure coordinates are within bounds
                x = max(0, x)
                y = max(0, y)
                w = min(iw - x, w)
                h = min(ih - y, h)

                if w > 0 and h > 0:
                    return x, y, w, h

        return None

    def process_frame(self, frame):
        """Process a frame to detect persons and faces"""
        # Calculate FPS
        self.new_frame_time = time.time()
        self.fps = 1 / (self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time

        # Create a copy of the frame for drawing
        output_frame = frame.copy()

        # Detect persons
        persons = self.detect_person(frame)

        # Process each detected person
        for bbox, person_img in persons:
            # Draw person bounding box
            x, y, w, h = bbox
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Detect face in the person crop
            face = self.detect_face(person_img)

            if face is not None:
                fx, fy, fw, fh = face
                # Convert face coordinates to original frame coordinates
                fx_abs = x + fx
                fy_abs = y + fy

                # Calculate dynamic padding based on face size
                # Use a percentage of face width and height (e.g., 50% padding)
                padding_percent = 0.5  # 50% padding
                # Enforce minimum padding (10px) for very small faces
                min_padding = 10

                # Calculate horizontal padding (x-direction)
                pad_x = max(int(fw * padding_percent), min_padding)
                # Calculate vertical padding (y-direction)
                pad_y = max(int(fh * padding_percent), min_padding)

                # Calculate padded box coordinates
                x1 = max(0, fx_abs - pad_x)
                y1 = max(0, fy_abs - pad_y)
                x2 = min(output_frame.shape[ 1 ], fx_abs + fw + pad_x)
                y2 = min(output_frame.shape[ 0 ], fy_abs + fh + pad_y)

                # Draw adaptively padded face bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display FPS
        cv2.putText(output_frame, f"FPS: {int(self.fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return output_frame

import cv2
import torch
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.neighbors import NearestNeighbors

from deep_sort_realtime.deepsort_tracker import DeepSort
# Assuming FaceWithPerson is defined elsewhere

class FaceRecognition:
    def __init__(self):
        self.face_detector = FaceWithPerson()
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.known_embeddings = [ ]
        self.known_names = [ ]

        try:
            with open("../Embeddings/embeddings.pkl", "rb") as f:
                self.known_embeddings, self.known_names = pickle.load(f)
            print(f"Loaded {len(self.known_names)} faces from embeddings file.")
            # Initialize the KNN only if we have embeddings
            if len(self.known_embeddings) > 0:
                self.knn = NearestNeighbors(n_neighbors=1)
                self.knn.fit(self.known_embeddings)
        except FileNotFoundError:
            print("No existing embeddings found.")
            self.knn = None

    def get_embedding(self, face_img):
        # Check if face_img is valid
        if face_img is None or face_img.size == 0:
            return None

        try:
            face = cv2.resize(face_img, (160, 160))
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                embedding = self.model(face_tensor)
            return embedding.squeeze().numpy()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def recognize_face(self, face_img):
        if self.knn is None or len(self.known_embeddings) == 0:
            return None

        embedding = self.get_embedding(face_img)
        if embedding is None:
            return None

        distances, indices = self.knn.kneighbors([ embedding ])
        print(distances[0][0])
        if distances[ 0 ][ 0 ] > 0.6:  # Return name only if distance is small enough
            return self.known_names[ indices[ 0 ][ 0 ] ]
        return None


def for_faceWithPerson():
    face_detector = FaceWithPerson()
    cap = cv2.VideoCapture(3)

    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame
            result_frame = face_detector.process_frame(frame)

            # Display the result
            cv2.imshow("Face Detection", result_frame)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


def face_recognition_func():
    face_recognition = FaceRecognition()
    face_detector = FaceWithPerson()
    cap = cv2.VideoCapture(3)

    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame
            result = face_detector.detect_face(frame)

            # Check if face was detected
            if result:
                x, y, w, h = result
                # Draw rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract face region for recognition
                face_roi = frame[ y:y + h, x:x + w ]

                # Recognize the face
                identity = face_recognition.recognize_face(face_roi)

                # Display identity text above the rectangle
                if identity:
                    cv2.putText(frame, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    cv2.putText(frame , "Unknown person" ,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2 )
            # Display the result
            cv2.imshow("Face Recognition", frame)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


class Persontracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30,
                      n_init=3,
                      nms_max_overlap=1.0,
                      max_cosine_distance=0.3,
                      nn_budget=None)
        self.tracks = None
    def updates_trackes(self, detections, frame):
        self.tracks = self.tracker.update_tracks(detections, frame=frame)
        return self.tracks



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    face_detector = FaceWithPerson()
    face_recognition = FaceRecognition()
    person_traker = Persontracker()
    skip_frame = 5
    count_frame = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        count_frame += 1
        if count_frame % skip_frame != 0 :
            continue
        persons = face_detector.detect_person(frame)
        print(face_detector.persons_tracking)
        tracks = person_traker.updates_trackes(face_detector.persons_tracking,frame)
        print(tracks)
        for track in tracks:
            print(f"Track ID: {track.track_id}")
            print(f"Track state: {track.state}")
            print(f"Track is confirmed: {track.is_confirmed()}")
            print(f"Track time since update: {track.time_since_update}")
            print(f"Track hits: {track.hits}")
            print(f"Track age: {track.age}")
            print(f"Track features: {track.features if hasattr(track, 'features') else 'N/A'}")
            print(f"Bounding box (LTWH): {track.to_ltwh()}")  # Left, Top, Width, Height
            print(f"Bounding box (LTRB): {track.to_ltrb()}")  # Left, Top, Right, Bottom
            print(f"Track confidence: {track.get_det_conf() if hasattr(track, 'get_det_conf') else 'N/A'}")
            print(f"Track class: {track.get_det_class() if hasattr(track, 'get_det_class') else 'N/A'}")

            print(f"Track's mean: {track.mean}")
            print(f"Track's covariance: {track.covariance}")
            print("-----------------------------------")

        for bbox, person in persons:
            person_x, person_y, person_width, person_height = bbox
            print(f"Person detected: {bbox}")

            # Draw rectangle for the detected person
            cv2.rectangle(frame, (person_x, person_y),
                          (person_x + person_width, person_y + person_height),
                          (0, 0, 255), 2)  # Red color for person boxes

            # Label the person
            cv2.putText(frame, "Person", (person_x, person_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Now detect faces within the person
            result = face_detector.detect_face(person)
            if result:
                x, y, w, h = result

                # Calculate global coordinates for the face
                global_x = person_x + x
                global_y = person_y + y

                # Draw rectangle around the detected face (green)
                cv2.rectangle(frame, (global_x, global_y),
                              (global_x + w, global_y + h),
                              (0, 255, 0), 2)

                # Extract face region for recognition
                face_roi = person[ y:y + h, x:x + w ]

                # Recognize the face
                identity = face_recognition.recognize_face(face_roi)

                # Display identity text above the rectangle
                if identity:
                    cv2.putText(frame, identity, (global_x, global_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

        # Display the result
        cv2.imshow("Face Recognition", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()