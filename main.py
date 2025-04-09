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
    with open("embeddings.pkl", "wb") as f:
        pickle.dump((known_embeddings, known_names), f)
    print("✅ Embeddings registered and saved.")


def recognize_faces():
    # Load embeddings
    with open("embeddings.pkl", "rb") as f:
        known_embeddings, known_names = pickle.load(f)
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(known_embeddings)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_crops = detect_face(frame)
        for face, (x, y, w, h) in face_crops:
            embedding = get_embedding(face)
            dist, idx = knn.kneighbors([embedding], n_neighbors=1)
            name = "Unknown"
            if dist[0][0] < 0.8:  # threshold
                name = known_names[idx[0][0]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true', help='Register faces from dataset')
    args = parser.parse_args()

    if args.register:
        dataset_path = "C:/Users/divya/OneDrive/Desktop/face-recog/Person_identify/Real Images"  # folder with student folders
        register_dataset(dataset_path)
    else:
        recognize_faces()
