import cv2

from ultralytics import YOLO
import numpy as np
cap = cv2.VideoCapture(0)
model = YOLO('Yoloweights/yolov8n.pt')
from Person_identify.faceAndPerson.FaceWithperson import FaceRecognition, FaceWithPerson
from Person_identify.Attendance.Attendence import attendence
from Person_identify.Frontend.main import VideoApp
face_recognition = FaceRecognition()
face_detector = FaceWithPerson()
Attendence = attendence()
detected ={}
count = 0
while cap.isOpened():
    ret , frame = cap.read()
    results = model.track(frame ,classes=[0], show=True)
    print(detected)
    if not count%15 ==0 :
        count+=1
        continue
    count+=1
    new_frame =[]
    # Loop through detected boxes and update mask
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates (xyxy format)
            mask = np.zeros(frame.shape[ :2 ], dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box.xyxy[ 0 ])
            track_id = box.id.int().cpu().tolist()
            print(track_id[0])
            # Draw filled rectangle for each detected person
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            # Apply mask to frame
            person = cv2.bitwise_and(frame, frame, mask=mask)
            new_frame.append(person)
            detect_face = face_detector.detect_face(person)
            identity = None

            if detect_face:
                x, y, w, h = detect_face
                face_crop = person[y:y + h, x:x + w]
                identity = face_recognition.recognize_face(face_crop)

            if identity:
                Attendence.take_attendence(identity)
                cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(frame, "Unknown person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            detected[identity] = track_id
            Attendence.get_attendence()
    Attendence.count+=1
    # Show the masked frame
    for i , frame_wise  in enumerate(new_frame):
        cv2.imshow(f'YOLO Masked Person{i}', frame_wise)
    cv2.imshow('YOLO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
