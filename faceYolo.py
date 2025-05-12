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


class FaceRecognitionApp:
    def __init__(self, root):
        self.dataset_path = None
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        self.model = None
        self.facemodel = None

        # Variables
        self.is_registering = False
        self.is_recognizing = False
        self.cap = None
        self.target_name = None
        self.zoom_factor = 2
        self.current_frame = None
        self.known_embeddings = [ ]
        self.known_names = [ ]

        # Performance optimization variables
        self.frame_skip = 2  # Process every Nth frame
        self.frame_count = 0
        self.detection_queue = queue.Queue(maxsize=1)
        self.results_queue = queue.Queue()  # Queue for detection results
        self.last_faces = [ ]  # Cache last detected faces
        self.last_face_time = time.time()
        self.face_detection_interval = 0.5  # Detect faces every X seconds

        # Attendance tracking variables
        self.last_attendance_time = defaultdict(int)  # Dictionary to track when each person last registered attendance
        self.attendance_interval = 3 # 5 minutes in seconds
        self.attendance_buffer = [ ]  # Buffer for attendance records before writing to disk
        self.attendance_write_count = 0
        self.write_interval = 5  # Write to CSV every N attendance records
        self.expected_hours = 1 # Default expected hours per day

        # Try to load existing embeddings (defer heavy loading)
        self.embeddings_loaded = False

        # Attendance DataFrame (will load on demand)
        self.attendance_df = None

        # Create widgets first to show UI quickly
        self.create_widgets()

        # Load data in background after UI is shown
        self.root.after(100, self.load_data_background)

    def load_data_background(self):
        """Load data in background to prevent UI freeze"""
        threading.Thread(target=self.load_data, daemon=True).start()

    def load_data(self):
        """Load face embeddings and attendance data"""

        try:
            with open("Embeddings/embeddings.pkl", "rb") as f:
                self.known_embeddings, self.known_names = pickle.load(f)
            print(f"Loaded {len(self.known_names)} faces from embeddings file.")
            self.embeddings_loaded = True
        except FileNotFoundError:
            print("No existing embeddings found.")

        # Try to load existing attendance records
        try:
            self.attendance_df = pd.read_csv("Attendance/Attendance.csv")
            print(f"Loaded attendance records with {len(self.attendance_df)} entries.")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # Create a new attendance DataFrame if none exists
            self.attendance_df = pd.DataFrame(columns=[ "Name", "Date", "Time", "Duration" ])
            print("Created new attendance record.")



    def create_widgets(self):
        # Main frames
        left_frame = tk.Frame(self.root, width=300, bg="#e0e0e0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        right_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Video display
        self.video_label = tk.Label(right_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel
        control_frame = tk.LabelFrame(left_frame, text="Controls", bg="#e0e0e0", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=10)

        tk.Button(control_frame, text="Exit", command=self.root.destroy,
                  bg="#F44336", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

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

        self.recognition_button = tk.Button(recognition_frame, text="Start Recognition",
                                            command=self.toggle_recognition,
                                            bg="#FF9800", fg="white", padx=10, pady=5)
        self.recognition_button.pack(fill=tk.X, pady=5)

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

        # Progress bar
        self.progress = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=280, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)

        # Attendance report buttons
        report_frame = tk.LabelFrame(left_frame, text="Attendance Reports", bg="#e0e0e0", padx=10, pady=10)
        report_frame.pack(fill=tk.X, pady=10)

        tk.Button(report_frame, text="View LOH Report", command=self.show_loh_report,
                  bg="#673AB7", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

        tk.Button(report_frame, text="Set Expected Hours", command=self.set_expected_hours,
                  bg="#00BCD4", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

        # Exit button
        tk.Button(left_frame, text="Exit", command=self.on_close,
                  bg="#607D8B", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=10)




    def update_status(self, message):
        """Thread-safe status update"""
        pass

    def select_dataset_folder(self):
        folder_path = filedialog.askdirectory(title="Select Dataset Folder")
        if folder_path:
            self.dataset_path = folder_path

    def start_registration(self):
        if not hasattr(self, 'dataset_path'):
            messagebox.showerror("Error", "Please select a dataset folder first.")
            return

        # Load model if not already loaded
        if self.model is None:

            self.model = InceptionResnetV1(pretrained='vggface2').eval()

        if self.facemodel is None:
            self.facemodel = YOLO('Yoloweights/best.pt')

        # Start registration in a separate thread
        threading.Thread(target=self.register_dataset, daemon=True).start()

    def register_dataset(self):
        self.is_registering = True

        total_folders = len([ f for f in os.listdir(self.dataset_path)
                              if os.path.isdir(os.path.join(self.dataset_path, f)) ])
        processed_folders = 0

        known_embeddings = [ ]
        known_names = [ ]

        for person in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person)
            if not os.path.isdir(person_folder):
                continue

            processed_folders += 1
            self.progress[ 'value' ] = (processed_folders / total_folders) * 100
            self.root.update_idletasks()

            # Process only a sample of images for faster registration
            img_files = [ f for f in os.listdir(person_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) ]
            sample_size =len(img_files) # Process at most 5 images per person

            for img_name in img_files[ :sample_size ]:
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
        with open("Embeddings/embeddings.pkl", "wb") as f:
            pickle.dump((known_embeddings, known_names), f)

        # Update internal variables
        self.known_embeddings = known_embeddings
        self.known_names = known_names
        self.embeddings_loaded = True
        self.progress[ 'value' ] = 0
        self.is_registering = False

    def toggle_recognition(self):
        if self.is_recognizing:
            self.stop_recognition()
            self.recognition_button.config(text="Start Recognition", bg="#FF9800")
        else:
            self.start_recognition()
            self.recognition_button.config(text="Stop Recognition", bg="#F44336")

    def start_recognition(self):
        if not self.embeddings_loaded or len(self.known_embeddings) == 0:
            messagebox.showerror("Error", "No face embeddings available. Please register faces first.")
            return

        # Load models if not already loaded
        if self.model is None:
            self.update_status("Loading face recognition model...")
            self.model = InceptionResnetV1(pretrained='vggface2').eval()

        if self.facemodel is None:
            self.update_status("Loading face detection model...")
            self.facemodel = YOLO('Yoloweights/best.pt')

        self.is_recognizing = True


        # Initialize KNN only once
        from sklearn.neighbors import NearestNeighbors
        self.knn = NearestNeighbors(n_neighbors=1)
        self.knn.fit(self.known_embeddings)

        # Open camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Start detection worker thread
        threading.Thread(target=self.detection_worker, daemon=True).start()

        # Start recognition thread (UI update)
        threading.Thread(target=self.recognition_loop, daemon=True).start()

    def detection_worker(self):
        """Worker thread to process face detection and recognition in background"""
        while self.is_recognizing:
            try:
                # Get frame from queue
                if not self.detection_queue.empty():
                    frame = self.detection_queue.get(block=False)

                    # Run face detection
                    face_crops = self.detect_face(frame)
                    recognized_faces = [ ]

                    # Process each detected face
                    for face, (x, y, w, h) in face_crops:
                        embedding = self.get_embedding(face)
                        dist, idx = self.knn.kneighbors([ embedding ], n_neighbors=1)
                        name = "Unknown"
                        attendance_recorded = False

                        if dist[ 0 ][ 0 ] < 0.8:
                            name = self.known_names[ idx[ 0 ][ 0 ] ]
                            # Record attendance (throttled)
                            attendance_recorded = self.take_attendance(name)

                        recognized_faces.append((name, (x, y, w, h), attendance_recorded))

                    # Put results in queue
                    if self.results_queue.full():
                        self.results_queue.get()  # Remove old result if queue is full
                    self.results_queue.put(recognized_faces)

                else:
                    time.sleep(0.01)  # Short sleep when queue is empty

            except Exception as e:
                print(f"Error in detection worker: {e}")
                time.sleep(0.1)

    def stop_recognition(self):
        self.is_recognizing = False

        # Flush attendance buffer
        self.flush_attendance_buffer()

        if self.cap:
            self.cap.release()
        self.update_status("Recognition stopped")

    def take_attendance(self, name):
        """Records attendance for a person, but only once every 5 minutes"""
        current_time = time.time()

        # Check if enough time has passed since last attendance record
        if current_time - self.last_attendance_time[ name ] >= self.attendance_interval:
            # Update last attendance time
            self.last_attendance_time[ name ] = int(current_time)

            # Get current date and time
            now = datetime.datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_time_str = now.strftime("%H:%M:%S")

            # Add to attendance buffer
            self.attendance_buffer.append({
                "Name": name,
                "Date": current_date,
                "Time": current_time_str,
                "Duration": 5  # 5 minutes attendance duration
            })

            # Increment write counter
            self.attendance_write_count += 1

            # Write to disk if buffer is full enough
            if self.attendance_write_count >= self.write_interval:
                self.flush_attendance_buffer()

            # Update status
            self.update_status(f"Attendance recorded for {name}")
            return True

        return False

    def flush_attendance_buffer(self):
        """Write buffered attendance records to disk"""
        if not self.attendance_buffer:
            return

        # Load existing data if needed
        if self.attendance_df is None:
            try:
                self.attendance_df = pd.read_csv("Attendance/Attendance.csv")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                self.attendance_df = pd.DataFrame(columns=[ "Name", "Date", "Time", "Duration" ])

        # Add buffer to dataframe
        new_records = pd.DataFrame(self.attendance_buffer)
        self.attendance_df = pd.concat([ self.attendance_df, new_records ], ignore_index=True)

        # Save to CSV file
        self.attendance_df.to_csv("Attendance.csv", index=False)

        # Clear buffer and reset counter
        self.attendance_buffer = [ ]
        self.attendance_write_count = 0

    def recognition_loop(self):
        """Update UI with camera feed and recognition results"""
        last_frame_time = time.time()
        frame_times = [ ]

        while self.is_recognizing:
            try:
                start_time = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    break

                # Frame counter for skipping frames
                self.frame_count += 1

                # Process only every Nth frame for detection
                if self.frame_count % self.frame_skip == 0 and time.time() - self.last_face_time >= self.face_detection_interval:
                    # Put frame in detection queue if it's empty
                    if self.detection_queue.empty():
                        self.detection_queue.put(frame.copy())
                        self.last_face_time = time.time()

                # Make a copy of the frame for drawing
                display_frame = frame.copy()

                # Apply zoom if target is set
                zoom_applied = False
                if self.target_name and self.last_faces:
                    for name, (x, y, w, h), _ in self.last_faces:
                        if name.lower() == self.target_name.lower():
                            # Calculate zoom and crop
                            try:
                                height, width = frame.shape[ :2 ]
                                # Center of the face
                                face_center_x = x + w // 2
                                face_center_y = y + h // 2

                                # Resize frame (zoom in)
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

                                cropped_zoom = zoomed_frame[ top:top + height, left:left + width ]
                                display_frame = cropped_zoom
                                zoom_applied = True
                                break
                            except Exception as e:
                                print(f"Zoom error: {e}")

                # Get and display face detection results
                if not self.results_queue.empty():
                    self.last_faces = self.results_queue.get()

                # If no zoom was applied, draw boxes for all detected faces
                if not zoom_applied and self.last_faces:
                    for name, (x, y, w, h), attendance_recorded in self.last_faces:
                        if name == "Unknown":
                            color = (0, 0, 255)  # Red for unknown
                        elif attendance_recorded:
                            color = (0, 255, 0)  # Green if attendance just recorded
                        else:
                            color = (0, 165, 255)  # Orange otherwise

                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(display_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Convert to RGB for tkinter (optimize by resizing first)
                display_height, display_width = display_frame.shape[ :2 ]

                # Calculate scaling for display
                video_width = self.video_label.winfo_width()
                video_height = self.video_label.winfo_height()

                if video_width > 1 and video_height > 1:  # Check if widget has been drawn
                    # Scale down frame for display to improve performance
                    scale = min(video_width / display_width, video_height / display_height)
                    if scale < 1:  # Only scale down, not up
                        new_width = int(display_width * scale)
                        new_height = int(display_height * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height),
                                                   interpolation=cv2.INTER_AREA)

                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                # Convert directly to PhotoImage without PIL for better performance
                img = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=img)

                # Update only if window still exists
                if self.is_recognizing:
                    self.video_label.config(image=photo)
                    self.video_label.image = photo  # Keep a reference

                # Calculate FPS
                current_time = time.time()
                frame_time = current_time - last_frame_time
                last_frame_time = current_time

                # Keep a running average of frame times for more stable FPS display
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)

                avg_frame_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

                # Update FPS counter every 10 frames


                # Throttle UI updates to not overload the system
                elapsed = time.time() - start_time
                if elapsed < 0.033:  # Aim for ~30 FPS max
                    time.sleep(0.033 - elapsed)

            except Exception as e:
                print(f"Error in recognition loop: {e}")
                time.sleep(0.1)

    def set_target_person(self):
        target = self.target_entry.get().strip()
        if not target:
            messagebox.showerror("Error", "Please enter a name")
            return

        self.target_name = target
        self.update_status(f"Target set to: {target}")

    def clear_target_person(self):
        self.target_name = None
        self.target_entry.delete(0, tk.END)
        self.update_status("Target cleared")

    def detect_face(self, frame):
        """
        Detect faces using YOLO model
        Returns a list of tuples: (face_crop, (x, y, w, h))
        """
        # Downsize image for faster detection if needed
        height, width = frame.shape[ :2 ]
        if width > 640:
            scale = 640 / width
            resized = cv2.resize(frame, (640, int(height * scale)), interpolation=cv2.INTER_AREA)
            scale_factor = width / 640
        else:
            resized = frame
            scale_factor = 1.0

        # Run detection with streamlined parameters
        results = self.facemodel(resized, verbose=False, conf=0.5, stream=True)
        face_crops = [ ]

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[ 0 ]
                # Scale back to original size
                x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(
                    y2 * scale_factor)
                w, h = x2 - x1, y2 - y1

                # Extract face crop (original frame)
                face_crop = frame[ y1:y2, x1:x2 ]

                # Skip if face is too small
                if face_crop.size == 0 or w < 20 or h < 20:
                    continue

                # Add to the list
                face_crops.append((face_crop, (x1, y1, w, h)))

        return face_crops

    def get_embedding(self, face_img):
        """Get face embedding with optimization for speed"""
        # Resize to 160x160 directly with cv2 (faster than PIL)
        face = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            embedding = self.model(face_tensor)

        return embedding.squeeze().numpy()

    def calculate_loh(self):
        """
        Calculate Loss of Hours (LOH) for each student
        Returns a DataFrame with student names and their LOH
        """
        # Flush any pending attendance records
        self.flush_attendance_buffer()

        # Load attendance data if needed
        if self.attendance_df is None or len(self.attendance_df) == 0:
            try:
                self.attendance_df = pd.read_csv("Attendance/Attendance.csv")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                return pd.DataFrame(columns=[ "Name", "Total Hours", "Expected Hours", "LOH" ])

        if len(self.attendance_df) == 0:
            return pd.DataFrame(columns=[ "Name", "Total Hours", "Expected Hours", "LOH" ])

        # Group by name and date, sum duration
        attendance_by_day = self.attendance_df.groupby([ 'Name', 'Date' ])[ 'Duration' ].sum().reset_index()

        # Group by name to get total hours
        total_hours = attendance_by_day.groupby('Name')[ 'Duration' ].sum().reset_index()
        # Convert minutes to hours properly
        total_hours[ 'Duration' ] = total_hours[ 'Duration' ]  # Convert minutes to hours

        # Count unique days for each person
        days_present = attendance_by_day.groupby('Name')[ 'Date' ].nunique().reset_index()
        days_present.rename(columns={'Date': 'Days Present'}, inplace=True)

        # Merge the two dataframes
        report = pd.merge(total_hours, days_present, on='Name')

        # Calculate expected hours
        report[ 'Expected Hours' ] = report[ 'Days Present' ] * self.expected_hours

        # Calculate LOH
        report[ 'LOH' ] = report[ 'Expected Hours' ] - report[ 'Duration' ]
        report[ 'LOH' ] = report[ 'LOH' ].apply(lambda x: max(0, x))  # LOH cannot be negative

        # Rename columns for clarity
        report.rename(columns={'Duration': 'Total Hours'}, inplace=True)

        return report[ [ 'Name', 'Total Hours', 'Expected Hours', 'LOH' ] ]
    def show_loh_report(self):
        """Show Loss of Hours report in a new window"""
        # Calculate LOH
        report = self.calculate_loh()

        if len(report) == 0:
            messagebox.showinfo("Report", "No attendance data available.")
            return

        # Create new window
        report_window = tk.Toplevel(self.root)
        report_window.title("Loss of Hours Report")
        report_window.geometry("600x400")

        # Create a frame for the report
        frame = tk.Frame(report_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create Treeview widget
        columns = ("Name", "Total Hours", "Expected Hours", "LOH")
        tree = ttk.Treeview(frame, columns=columns, show="headings")

        # Set column headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)

        # Insert data
        for _, row in report.iterrows():
            tree.insert("", tk.END, values=(
                row[ 'Name' ],
                f"{row[ 'Total Hours' ]:.2f}",
                f"{row[ 'Expected Hours' ]:.2f}",
                f"{row[ 'LOH' ]:.2f}"
            ))

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add export button
        export_frame = tk.Frame(report_window)
        export_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(export_frame, text="Export to CSV", command=lambda: self.export_report(report),
                  bg="#4CAF50", fg="white", padx=10, pady=5).pack(side=tk.RIGHT)

    def export_report(self, report):
        """Export LOH report to CSV"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[ ("CSV files", "*.csv"), ("All files", "*.*") ],
            title="Save Report As"
        )

        if file_path:
            report.to_csv(file_path, index=False)
            messagebox.showinfo("Export", f"Report exported successfully to {file_path}")

    def set_expected_hours(self):
        """Set expected hours per day"""
        # Create simple dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Expected Hours")
        dialog.geometry("300x150")
        dialog.resizable(False, False)

        # Center on parent
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Hours per day:").pack(pady=(20, 5))

        hours_var = tk.StringVar(value=str(self.expected_hours))
        entry = tk.Entry(dialog, textvariable=hours_var, width=10)
        entry.pack(pady=5)

        def save_hours():
            try:
                hours = float(hours_var.get())
                if hours <= 0:
                    messagebox.showerror("Error", "Hours must be greater than zero")
                    return
                self.expected_hours = hours
                dialog.destroy()
                messagebox.showinfo("Success", f"Expected hours set to {hours} per day")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")

        tk.Button(dialog, text="Save", command=save_hours,
                  bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=20)

        entry.focus_set()

    def on_close(self):
        """Handle application close"""
        if self.is_recognizing:
            self.stop_recognition()

        # Flush any pending attendance records
        self.flush_attendance_buffer()

        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)

    # Set window close handler
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.mainloop()