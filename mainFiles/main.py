#Main File
import cv2
import os
import numpy as np
import torch
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import queue
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import datetime
import pandas as pd
import time
from collections import defaultdict
from Person_identify.faceAndPerson import FaceWithperson

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        # Model setup
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

        # Performance settings - adjusted for better performance
        self.process_every_n_frames = 5
        self.frame_count = 0
        self.detection_size = (256, 192)
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.raw_frame_queue = queue.Queue(maxsize=5)  # Queue for raw frames

        # Variables
        self.is_registering = False
        self.is_recognizing = False
        self.cap = None
        self.target_name = None
        self.zoom_factor = 2
        self.current_frame = None
        self.known_embeddings = [ ]
        self.known_names = [ ]
        self.facemodel = None  # Lazy load YOLO model
        self.latest_faces = [ ]  # Store face data between frames
        self.attendance_total = 1
        self.attendance_details = {}
        self.attendance= True
        # Shared target face coordinates for both displays
        self.target_face_coords = None
        # In __init__ method, after other initializations
        self.face_person_detector = FaceWithperson.FaceWithPerson()
        self.face_person_detector.start_detection_thread()
        # Attendance tracking variables

        self.attendance_interval = 10 # 60 seconds between attendance records
        self.attendance_records = [ ]
        self.last_attendance_time = defaultdict(float)  # Use float for timestamps
        self.global_last_attendance_time = 0
        # Try to load existing embeddings
        try:
            with open("../Embeddings/embeddings.pkl", "rb") as f:
                self.known_embeddings, self.known_names = pickle.load(f)
            print(f"Loaded {len(self.known_names)} faces from embeddings file.")
        except FileNotFoundError:
            print("No existing embeddings found.")

        # Try to load existing attendance records
        try:
            self.attendance_df = pd.read_csv("./Attendance/Attendance.csv")
            print(f"Loaded attendance records with {len(self.attendance_df)} entries.")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # Create a new attendance DataFrame if none exists
            self.attendance_df = pd.DataFrame(columns=[ "Name", "Date", "Time", "Duration" ])
            print("Created new attendance record.")

        self.create_widgets()

        # Start update display loops (separate from recognition)
        threading.Thread(target=self.update_detection_display_loop, daemon=True).start()
        threading.Thread(target=self.update_raw_display_loop, daemon=True).start()

    def create_widgets(self):
        # Main frames
        left_frame = tk.Frame(self.root, width=300, bg="#e0e0e0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        right_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create two video frames in the right panel
        video_frame = tk.Frame(right_frame, bg="#f0f0f0")
        video_frame.pack(fill=tk.BOTH, expand=True)

        # Top video frame (detection)
        detection_frame = tk.LabelFrame(video_frame, text="Face Detection", bg="#e0e0e0", padx=5, pady=5)
        detection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.detection_label = tk.Label(detection_frame, bg="black")
        self.detection_label.pack(fill=tk.BOTH, expand=True)

        # Bottom video frame (raw video)
        raw_frame = tk.LabelFrame(video_frame, text="Raw Video", bg="#e0e0e0", padx=5, pady=5)
        raw_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.raw_video_label = tk.Label(raw_frame, bg="black")
        self.raw_video_label.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = tk.LabelFrame(left_frame, text="Controls", bg="#e0e0e0", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=10)

        # Performance settings
        perf_frame = tk.LabelFrame(control_frame, text="Performance", bg="#e0e0e0", padx=10, pady=5)
        perf_frame.pack(fill=tk.X, pady=5)

        tk.Label(perf_frame, text="Process every N frames:", bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
        self.frame_skip_var = tk.StringVar(value=str(self.process_every_n_frames))
        frame_skip_entry = tk.Entry(perf_frame, textvariable=self.frame_skip_var, width=5)
        frame_skip_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(perf_frame, text="Apply",
                  command=lambda: self.set_frame_skip(int(self.frame_skip_var.get())),
                  bg="#4CAF50", fg="white", padx=5, pady=2).pack(side=tk.LEFT, padx=5)

        # Registration section
        register_frame = tk.LabelFrame(left_frame, text="Registration", bg="#e0e0e0", padx=10, pady=10)
        register_frame.pack(fill=tk.X, pady=10)

        tk.Button(register_frame, text="Select Dataset Folder", command=self.select_dataset_folder,
                  bg="#4CAF50", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

        tk.Button(register_frame, text="Register Faces", command=self.start_registration,
                  bg="#2196F3", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

        # Recognition section
        recognition_frame = tk.LabelFrame(left_frame, text="Recognition", bg="#e0e0e0", padx=10, pady=10)
        recognition_frame.pack(fill=tk.X, pady=10)

        self.recognition_btn = tk.Button(recognition_frame, text="Start Recognition", command=self.toggle_recognition,
                                         bg="#FF9800", fg="white", padx=10, pady=5)
        self.recognition_btn.pack(fill=tk.X, pady=5)

        # Target person frame
        target_frame = tk.LabelFrame(left_frame, text="Target Person", bg="#e0e0e0", padx=10, pady=10)
        target_frame.pack(fill=tk.X, pady=10)

        tk.Label(target_frame, text="Focus on person:", bg="#e0e0e0").pack(anchor=tk.W)

        self.target_entry = tk.Entry(target_frame)
        self.target_entry.pack(fill=tk.X, pady=5)

        tk.Button(target_frame, text="Set Target", command=self.set_target_person,
                  bg="#9C27B0", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

        tk.Button(target_frame, text="Clear Target", command=self.clear_target_person,
                  bg="#F44336", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

        # Status section

        # Attendance view button
        tk.Button(left_frame, text="View Attendance Records", command=self.view_attendance_records,
                  bg="#673AB7", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

        # Exit button
        tk.Button(left_frame, text="Exit", command=self.on_close,
                  bg="#607D8B", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=10)

    def update_status(self, message):
        """Update status label - thread safe"""
        #self.root.after(0, lambda: self.status_label.config(text=message))
        pass

    def set_frame_skip(self, n):
        """Set the frame skip value"""
        if n < 1:
            n = 1
        self.process_every_n_frames = n
        self.update_status(f"Processing every {n} frames")

    def select_dataset_folder(self):
        folder_path = filedialog.askdirectory(title="Select Dataset Folder")
        if folder_path:
            self.dataset_path = folder_path
            self.update_status(f"Dataset folder selected: {os.path.basename(folder_path)}")

    def start_registration(self):
        if not hasattr(self, 'dataset_path'):
            messagebox.showerror("Error", "Please select a dataset folder first.")
            return

        # Start registration in a separate thread
        threading.Thread(target=self.register_dataset, daemon=True).start()

    def load_yolo_model(self):
        """Lazy loading of YOLO model to improve startup time"""
        if self.facemodel is None:
            self.update_status("Loading YOLO model...")
            self.facemodel = YOLO('../Yoloweights/best.pt')
            self.update_status("YOLO model loaded")

    def register_dataset(self):
        self.update_status("Registration started...")
        self.is_registering = True

        # Ensure YOLO model is loaded
        self.load_yolo_model()

        total_folders = len([ f for f in os.listdir(self.dataset_path)
                              if os.path.isdir(os.path.join(self.dataset_path, f)) ])
        processed_folders = 0

        known_embeddings = [ ]
        known_names = [ ]

        for person in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person)
            self.update_status(f"Processing: {person}")

            if not os.path.isdir(person_folder):
                continue

            processed_folders += 1
            #self.root.after(0, lambda val=(processed_folders / total_folders) * 100: self.progress.config(value=val))

            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                face_crops = self.detect_face(img)

                if face_crops:
                    face, _ = face_crops[ 0 ]
                    embedding = self.get_embedding(face)
                    known_embeddings.append(embedding)
                    known_names.append(person)

        # Save embeddings
        with open("../Embeddings/embeddings.pkl", "wb") as f:
            pickle.dump((known_embeddings, known_names), f)

        # Update internal variables
        self.known_embeddings = known_embeddings
        self.known_names = known_names

        #self.root.after(0, lambda: self.people_label.config(text=f"People in database: {len(set(known_names))}"))
        self.update_status(f"Registration complete. {len(known_names)} faces registered.")
        #self.root.after(0, lambda: self.progress.config(value=0))
        self.is_registering = False

    def toggle_recognition(self):
        if self.is_recognizing:
            self.stop_recognition()
            self.recognition_btn.config(text="Start Recognition")
        else:
            self.start_recognition()
            self.recognition_btn.config(text="Stop Recognition")

    def start_recognition(self):
        if len(self.known_embeddings) == 0:
            messagebox.showerror("Error", "No face embeddings available. Please register faces first.")
            return

        # Ensure YOLO model is loaded
        self.load_yolo_model()

        self.is_recognizing = True
        self.update_status("Recognition started")

        # Initialize KNN
        from sklearn.neighbors import NearestNeighbors
        self.knn = NearestNeighbors(n_neighbors=1)
        self.knn.fit(self.known_embeddings)

        # Open camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Start frame capture thread
        threading.Thread(target=self.capture_frames, daemon=True).start()

        # Start recognition thread
        threading.Thread(target=self.process_frames, daemon=True).start()

    def stop_recognition(self):
        self.is_recognizing = False
        if self.cap:
            self.cap.release()
        self.update_status("Recognition stopped")

    def capture_frames(self):
        """Thread to capture frames from camera"""
        last_time = time.time()
        frame_count = 0

        while self.is_recognizing:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed > 1.0:  # Update FPS every second
                fps = frame_count / elapsed
                #self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {fps:.1f}"))
                last_time = current_time
                frame_count = 0
                print(fps)
            # Make a copy of the frame for raw video feed
            try:
                self.raw_frame_queue.put(frame.copy(), block=False)
            except queue.Full:
                pass

            # Add frame to processing queue
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass


    def process_frames(self):
        """Thread to process frames"""
        frame_counter = 0

        while self.is_recognizing:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)

                # Skip frames based on setting
                frame_counter += 1
                if frame_counter % self.process_every_n_frames != 0:
                    # For skipped frames, put previous results in result queue
                    if self.latest_faces:
                        try:
                            self.result_queue.put((frame, self.latest_faces), block=False)
                        except queue.Full:
                            pass
                    continue

                # Process frame for face detection & recognition
                height, width = frame.shape[ :2 ]

                # Resize for detection (smaller is faster)
                detection_frame = cv2.resize(frame, self.detection_size)

                # Detect faces
                # Use the FaceWithPerson detector instead of detect_face
                result_frame = self.face_person_detector.process_frame(detection_frame)
                face_crops = [ ]

                # Get faces detected by the face_person_detector
                for x, y, w, h in self.face_person_detector.last_faces:
                    # Scale coordinates back to original frame size
                    orig_x = int(x * (width / self.detection_size[ 0 ]))
                    orig_y = int(y * (height / self.detection_size[ 1 ]))
                    orig_w = int(w * (width / self.detection_size[ 0 ]))
                    orig_h = int(h * (height / self.detection_size[ 1 ]))

                    # Get face from detection frame
                    if (x >= 0 and y >= 0 and x + w <= detection_frame.shape[ 1 ] and y + h <= detection_frame.shape[
                        0 ]):
                        face = detection_frame[ y:y + h, x:x + w ]
                        face_crops.append((face, (x, y, w, h)))

                processed_faces = [ ]

                if face_crops:
                    # Prepare batch for face recognition
                    face_tensors = [ ]
                    face_coords = [ ]

                    for face, (x, y, w, h) in face_crops:
                        # Scale coordinates back to original frame
                        orig_x = int(x * (width / self.detection_size[ 0 ]))
                        orig_y = int(y * (height / self.detection_size[ 1 ]))
                        orig_w = int(w * (width / self.detection_size[ 0 ]))
                        orig_h = int(h * (height / self.detection_size[ 1 ]))

                        # Get face from original frame for better quality
                        if (orig_x >= 0 and orig_y >= 0 and
                                orig_x + orig_w <= width and orig_y + orig_h <= height):
                            face = frame[ orig_y:orig_y + orig_h, orig_x:orig_x + orig_w ]

                            # Prepare for batch processing
                            face_resized = cv2.resize(face, (160, 160))
                            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                            face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).float() / 255.0
                            face_tensors.append(face_tensor)
                            face_coords.append((orig_x, orig_y, orig_w, orig_h))

                    if face_tensors:
                        # Batch process faces for embeddings
                        with torch.no_grad():
                            face_batch = torch.stack(face_tensors)
                            embeddings = self.model(face_batch)

                        # Get nearest neighbors for all embeddings at once
                        if len(embeddings) > 0:
                            distances, indices = self.knn.kneighbors(embeddings.numpy())

                            # Process results
                            for i, ((x, y, w, h), dist, idx) in enumerate(zip(face_coords, distances, indices)):
                                name = "Unknown"
                                if dist[ 0 ] < 0.8:
                                    name = self.known_names[ idx[ 0 ] ]
                                    # Record attendance
                                    if self.attendance:
                                        self.take_attendance(name)

                                processed_faces.append((name, (x, y, w, h), dist[ 0 ] < 0.8))
                            # Replace the problematic attendance timing section with:
                            # Replace this section in process_frames():


                if self.attendance:
                    # If in attendance mode, record attendance
                    if processed_faces:
                        self.attendance = False
                        self.global_last_attendance_time = time.time()
                elif time.time() - self.global_last_attendance_time >= self.attendance_interval:
                    # If enough time has passed since last attendance check, enable it again
                    self.attendance_total += 1
                    self.attendance = True
                # Find target person's face coordinates
                self.target_face_coords = None
                if self.target_name:
                    for name, (x, y, w, h), is_known in processed_faces:
                        if is_known and name.lower() == self.target_name.lower():
                            self.target_face_coords = (x, y, w, h)
                            break

                # Store latest faces for use with skipped frames
                self.latest_faces = processed_faces

                # Add processed frame to result queue
                try:
                    self.result_queue.put((frame, processed_faces), block=False)
                except queue.Full:
                    pass

            except queue.Empty:
                time.sleep(0.01)  # Short sleep to avoid CPU spinning
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)



    def update_detection_display_loop(self):
        """Thread to update the detection display - separate from processing"""
        last_update_time = time.time()

        while True:
            try:
                # Get processed frame from queue
                frame, faces = self.result_queue.get(timeout=0.1)

                # Make a copy for drawing
                display_frame = frame.copy()

                zoom_applied = False

                # Handle target person zoom
                if self.target_name and self.target_face_coords:
                    x, y, w, h = self.target_face_coords
                    # Center of the face
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2

                    # Calculate zoom
                    height, width = frame.shape[ :2 ]

                    # Resize whole frame (zoom in)
                    zoomed_frame = cv2.resize(frame, None, fx=self.zoom_factor, fy=self.zoom_factor)
                    zh, zw = zoomed_frame.shape[ :2 ]

                    # Calculate the crop region centered on the face
                    center_x_zoomed = int(face_center_x * self.zoom_factor)
                    center_y_zoomed = int(face_center_y * self.zoom_factor)

                    left = center_x_zoomed - width // 2
                    top = center_y_zoomed - height // 2

                    # Clip to bounds
                    left = max(0, min(zw - width, left))
                    top = max(0, min(zh - height, top))

                    # Get the zoomed and cropped area
                    if top + height <= zh and left + width <= zw:
                        cropped_zoom = zoomed_frame[ top:top + height, left:left + width ]
                        display_frame = cropped_zoom
                        zoom_applied = True

                # Draw boxes for recognized faces if not zoomed
                if not zoom_applied:
                    for name, (x, y, w, h), is_known in faces:
                        color = (0, 255, 0) if is_known else (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(display_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Update the detection display
                self.update_detection_display(display_frame)

                # Control update rate
                current_time = time.time()
                elapsed = current_time - last_update_time
                if elapsed < 0.033:  # Cap at ~30 FPS for display
                    time.sleep(0.033 - elapsed)
                last_update_time = time.time()

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error updating detection display: {e}")
                time.sleep(0.1)

    def update_raw_display_loop(self):
        """Thread to update the raw video display"""
        last_update_time = time.time()

        while True:
            try:
                # Get raw frame from queue
                raw_frame = self.raw_frame_queue.get(timeout=0.1)

                # Apply the same zoom effect for target person if needed
                if self.target_name and self.target_face_coords:
                    x, y, w, h = self.target_face_coords
                    # Center of the face
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2

                    # Calculate zoom
                    height, width = raw_frame.shape[ :2 ]

                    # Resize whole frame (zoom in)
                    zoomed_frame = cv2.resize(raw_frame, None, fx=self.zoom_factor, fy=self.zoom_factor)
                    zh, zw = zoomed_frame.shape[ :2 ]

                    # Calculate the crop region centered on the face
                    center_x_zoomed = int(face_center_x * self.zoom_factor)
                    center_y_zoomed = int(face_center_y * self.zoom_factor)

                    left = center_x_zoomed - width // 2
                    top = center_y_zoomed - height // 2

                    # Clip to bounds
                    left = max(0, min(zw - width, left))
                    top = max(0, min(zh - height, top))

                    # Get the zoomed and cropped area
                    if top + height <= zh and left + width <= zw:
                        raw_frame = zoomed_frame[ top:top + height, left:left + width ]

                # Update the raw video display
                self.update_raw_display(raw_frame)

                # Control update rate
                current_time = time.time()
                elapsed = current_time - last_update_time
                if elapsed < 0.033:  # Cap at ~30 FPS for display
                    time.sleep(0.033 - elapsed)
                last_update_time = time.time()

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error updating raw video display: {e}")
                time.sleep(0.1)

    def update_detection_display(self, frame):
        """Update the detection display with the given frame"""
        # Convert to RGB for tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use PIL for efficient image conversion
        img = Image.fromarray(rgb_frame)

        # Resize to fit the window if needed
        video_width = self.detection_label.winfo_width()
        video_height = self.detection_label.winfo_height()

        if video_width > 1 and video_height > 1:
            img_width, img_height = img.size
            scale = min(video_width / img_width, video_height / img_height)

            # Only resize if needed
            if scale < 1:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=img)

        # Update the label
        self.detection_label.config(image=photo)
        self.detection_label.image = photo  # Keep a reference

    def update_raw_display(self, frame):
        """Update the raw video display with the given frame"""
        # Convert to RGB for tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use PIL for efficient image conversion
        img = Image.fromarray(rgb_frame)

        # Resize to fit the window if needed
        video_width = self.raw_video_label.winfo_width()
        video_height = self.raw_video_label.winfo_height()

        if video_width > 1 and video_height > 1:
            img_width, img_height = img.size
            scale = min(video_width / img_width, video_height / img_height)

            # Only resize if needed
            if scale < 1:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=img)

        # Update the label
        self.raw_video_label.config(image=photo)
        self.raw_video_label.image = photo  # Keep a reference

    def take_attendance(self, name):
        """Records attendance for a person, but only once every X seconds"""
        current_time = time.time()

        # Check if enough time has passed since last attendance record
        if current_time - self.last_attendance_time[ name ] >= self.attendance_interval:
            # Update last attendance time
            self.last_attendance_time[ name ] = current_time

            # Get current date and time
            now = datetime.datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_time_str = now.strftime("%H:%M:%S")

            # Add to attendance dataframe
            new_record = pd.DataFrame({
                "Name": [ name ],
                "Date": [ current_date ],
                "Time": [ current_time_str ],
                "Duration": [ 5 ]  # 5 minutes attendance duration
            })

            self.attendance_df = pd.concat([ self.attendance_df, new_record ], ignore_index=True)
            self.attendance_details[ name ] = self.attendance_details.get(name, 0) + 1
            # Save to CSV file - do this less frequently to avoid file I/O overhead
            self.attendance_df.to_csv("./attendance/Attendance.csv", index=False)

            # Update status
            self.update_status(f"Attendance recorded for {name}")
            return True

        return False
    def view_attendance_records(self):
        """Display a window showing attendance records"""
        if len(self.attendance_df) == 0:
            messagebox.showinfo("No Data", "No attendance records found.")
            return

        # Create a new window
        print(self.attendance_details)
        print(self.attendance_total)

        # For calculating percentages
        for name, count in self.attendance_details.items():
            percentage = (float(count) / self.attendance_total) * 100
            print(f'{name} : {percentage:.2f}%')
        report_window = tk.Toplevel(self.root)
        report_window.title("Attendance Records")
        report_window.geometry("600x400")


        frame = tk.Frame(report_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a treeview (table)
        tree = ttk.Treeview(frame)
        tree[ "columns" ] = list(self.attendance_df.columns)
        tree[ "show" ] = "headings"

        # Define column headings
        for col in self.attendance_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)

        # Add data to the table (limit to last 100 entries for performance)
        display_df = self.attendance_df.tail(100)
        for _, row in display_df.iterrows():
            tree.insert("", tk.END, values=list(row))

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)

    def set_target_person(self):
        target = self.target_entry.get().strip()
        if not target:
            messagebox.showerror("Error", "Please enter a name")
            return

        self.target_name = target
        self.update_status(f"Target set to: {target}")

    def clear_target_person(self):
        self.target_name = None
        self.target_face_coords = None
        self.target_entry.delete(0, tk.END)
        self.update_status("Target cleared")

    def detect_face(self, frame):
        """
        Detect faces using YOLO model
        Returns a list of tuples: (face_crop, (x, y, w, h))
        """
        # Performance optimization: limit processing
        results = self.facemodel(frame, verbose=False, conf=0.5, stream=True)
        face_crops = [ ]

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[ 0 ]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Extract face crop
                face_crop = frame[ y1:y2, x1:x2 ]

                # Add to the list
                face_crops.append((face_crop, (x1, y1, w, h)))

            return face_crops

    def get_embedding(self, face_img):
        """Get face embedding using the FaceNet model"""
        face = cv2.resize(face_img, (160, 160))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding.squeeze().numpy()

    def on_close(self):
        self.is_recognizing = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()