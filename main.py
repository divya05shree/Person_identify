import cv2
import os
import numpy as np
import torch
import pickle
from PIL import Image

from facenet_pytorch import InceptionResnetV1
from sklearn.neighbors import NearestNeighbors

extractor = InceptionResnetV1(pretrained='vggface2').eval()
model = InceptionResnetV1(pretrained='vggface2').eval()

model.eval()

def preprocess_face(cv2_img):
    face_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    inputs = extractor(images=pil_img, return_tensors="pt")
    return inputs

def get_embedding(face_img):
    face = cv2.resize(face_img, (160, 160))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.squeeze().numpy()


def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_crops = []
    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        face_crops.append((face_crop, (x, y, w, h)))
    return face_crops

def register_dataset(dataset_path):
    known_embeddings = []
    known_names = []

    for person in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person)
        print(f"/n📁 Entering folder: {person_folder}")

        if not os.path.isdir(person_folder):
            print("  ❌ Not a folder, skipping.")
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            print(f"  🖼️ Processing image: {img_path}")
            img = cv2.imread(img_path)

            if img is None:
                print("    ❌ Failed to load image!")
                continue

            face_crops = detect_face(img)
            print(f"    🔍 {len(face_crops)} face(s) detected")

            if face_crops:
                face, _ = face_crops[0]
                embedding = get_embedding(face)
                known_embeddings.append(embedding)
                known_names.append(person)
            else:
                print("    ⚠️ No face detected.")

    # Save only after all embeddings are collected
    with open("Embeddings/embeddings.pkl", "wb") as f:
        pickle.dump((known_embeddings, known_names), f)
    print("✅ Embeddings registered and saved.")


def recognize_faces():
    import cv2
    import pickle
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # Load embeddings
    with open("Embeddings/embeddings.pkl", "rb") as f:
        known_embeddings, known_names = pickle.load(f)
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(known_embeddings)

    cap = cv2.VideoCapture(0)
    target_name = None
    zoom_factor = 2 # 20% zoom

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        face_crops = detect_face(frame)  # [(face_img, (x, y, w, h)), ...]
        zoom_applied = False

        for face, (x, y, w, h) in face_crops:
            embedding = get_embedding(face)
            dist, idx = knn.kneighbors([embedding], n_neighbors=1)
            name = "Unknown"
            if dist[0][0] < 0.8:
                name = known_names[idx[0][0]]

            if target_name and name.lower() == target_name.lower():
                # Center of the face
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Resize whole frame (zoom in)
                zoomed_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)
                zh, zw = zoomed_frame.shape[:2]

                # Calculate the crop region centered on the face
                center_x_zoomed = int(face_center_x * zoom_factor)
                center_y_zoomed = int(face_center_y * zoom_factor)

                left = center_x_zoomed - width // 2
                top = center_y_zoomed - height // 2

                # Clip to bounds
                left = max(0, min(zw - width, left))
                top = max(0, min(zh - height, top))

                cropped_zoom = zoomed_frame[top:top + height, left:left + width]
                cv2.imshow("Real-Time Face Recognition", cropped_zoom)
                zoom_applied = True
                break

        if not zoom_applied:
            # Show normal frame with boxes and names
            for face, (x, y, w, h) in face_crops:
                embedding = get_embedding(face)
                dist, idx = knn.kneighbors([embedding], n_neighbors=1)
                name = "Unknown"
                if dist[0][0] < 0.8:
                    name = known_names[idx[0][0]]

                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("Real-Time Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            target_name = input("Enter name of the person to center and zoom: ").strip()
        elif key == ord('c'):
            target_name = None
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true', help='Register faces from dataset')
    args = parser.parse_args()

    if args.register:
        dataset_path = "Real Images"  # folder with student folders
        register_dataset(dataset_path)

    else:
        recognize_faces()
