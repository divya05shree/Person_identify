import cv2
import os
import numpy as np
import torch
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import datetime
import pandas as pd
import time
from collections import defaultdict


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        # Model setup
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

        # Variables
        self.is_registering = False
        self.is_recognizing = False
        self.cap = None
        self.target_name = None
        self.zoom_factor = 2
        self.current_frame = None
        self.known_embeddings = [ ]
        self.known_names = [ ]
        self.facemodel = YOLO('Yoloweights/best.pt')

        # Attendance tracking variables
        self.last_attendance_time = defaultdict(int)  # Dictionary to track when each person last registered attendance
        self.attendance_interval = 3  # 5 minutes in seconds
        self.attendance_records = [ ]  # List to store attendance records
        self.expected_hours = 8  # Default expected hours per day

        # Try to load existing embeddings
        try:
            with open("Embeddings/embeddings.pkl", "rb") as f:
                self.known_embeddings, self.known_names = pickle.load(f)
            print(f"Loaded {len(self.known_names)} faces from embeddings file.")
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

        self.create_widgets()

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

        tk.Button(recognition_frame, text="Start Recognition", command=self.toggle_recognition,
                  bg="#FF9800", fg="white", padx=10, pady=5).pack(fill=tk.X, pady=5)

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
        status_frame = tk.LabelFrame(left_frame, text="Status", bg="#e0e0e0", padx=10, pady=10)
        status_frame.pack(fill=tk.X, pady=10)

        self.status_label = tk.Label(status_frame, text="Ready", bg="#e0e0e0", wraplength=280, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X)

        # People count
        self.people_label = tk.Label(status_frame, text=f"People in database: {len(self.known_names)}",
                                     bg="#e0e0e0")
        self.people_label.pack(fill=tk.X, pady=5)

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
        self.status_label.config(text=message)
        self.root.update_idletasks()

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

    def register_dataset(self):
        self.update_status("Registration started...")
        self.is_registering = True

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
            self.progress[ 'value' ] = (processed_folders / total_folders) * 100
            self.root.update_idletasks()

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
        with open("Embeddings/embeddings.pkl", "wb") as f:
            pickle.dump((known_embeddings, known_names), f)

        # Update internal variables
        self.known_embeddings = known_embeddings
        self.known_names = known_names

        self.people_label.config(text=f"People in database: {len(set(known_names))}")
        self.update_status(f"Registration complete. {len(known_names)} faces registered.")
        self.progress[ 'value' ] = 0
        self.is_registering = False

    def toggle_recognition(self):
        if self.is_recognizing:
            self.stop_recognition()
        else:
            self.start_recognition()

    def start_recognition(self):
        if len(self.known_embeddings) == 0:
            messagebox.showerror("Error", "No face embeddings available. Please register faces first.")
            return

        self.is_recognizing = True
        self.update_status("Recognition started")

        # Initialize KNN
        from sklearn.neighbors import NearestNeighbors
        self.knn = NearestNeighbors(n_neighbors=1)
        self.knn.fit(self.known_embeddings)

        # Open camera
        self.cap = cv2.VideoCapture(0)

        # Start recognition thread
        threading.Thread(target=self.recognition_loop, daemon=True).start()

    def stop_recognition(self):
        self.is_recognizing = False
        if self.cap:
            self.cap.release()
        self.update_status("Recognition stopped")

    def take_attendance(self, name):
        """Records attendance for a person, but only once every 5 minutes"""
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

            # Save to CSV file
            self.attendance_df.to_csv("Attendance.csv", index=False)

            # Update status
            self.update_status(f"Attendance recorded for {name}")
            return True

        return False

    def recognition_loop(self):
        while self.is_recognizing:
            ret, frame = self.cap.read()
            if not ret:
                break

            height, width = frame.shape[ :2 ]
            face_crops = self.detect_face(frame)
            zoom_applied = False

            # Make a copy of the frame for drawing
            display_frame = frame.copy()

            if self.target_name and face_crops:
                # Check if target person is in frame
                for face, (x, y, w, h) in face_crops:
                    embedding = self.get_embedding(face)
                    dist, idx = self.knn.kneighbors([ embedding ], n_neighbors=1)
                    name = "Unknown"
                    if dist[ 0 ][ 0 ] < 0.8:
                        name = self.known_names[ idx[ 0 ][ 0 ] ]
                    try:
                        if name.lower() == self.target_name.lower():
                            # Center of the face
                            face_center_x = x + w // 2
                            face_center_y = y + h // 2

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

                            cropped_zoom = zoomed_frame[ top:top + height, left:left + width ]
                            display_frame = cropped_zoom
                            zoom_applied = True
                            break
                    except Exception as e:
                        print(e)
            if not zoom_applied:
                # Draw boxes and names for all faces
                for face, (x, y, w, h) in face_crops:
                    embedding = self.get_embedding(face)
                    dist, idx = self.knn.kneighbors([ embedding ], n_neighbors=1)
                    name = "Unknown"
                    if dist[ 0 ][ 0 ] < 0.8:
                        name = self.known_names[ idx[ 0 ][ 0 ] ]
                        # Record attendance
                        attendance_recorded = self.take_attendance(name)
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)  # Red for unknown

                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert to RGB for tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            # Resize to fit the window
            img_width, img_height = img.size
            video_width = self.video_label.winfo_width()
            video_height = self.video_label.winfo_height()

            if video_width > 1 and video_height > 1:  # Check if the widget has been drawn
                scale = min(video_width / img_width, video_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=img)

            # Update the label
            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep a reference

            # Update window
            self.root.update()

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
        height, width = frame.shape[ :2 ]
        if width > 640:
            scale = 640 / width
            resized = cv2.resize(frame, (640, int(height * scale)), interpolation=cv2.INTER_AREA)
            scale_factor = width / 640
        else:
            resized = frame
            scale_factor = 1.0

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
        face = cv2.resize(face_img, (160, 160),interpolation=cv2.INTER_AREA)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding.squeeze().numpy()

    def calculate_loh(self):
        """
        Calculate Loss of Hours (LOH) for each student
        Returns a DataFrame with student names and their LOH
        """
        if len(self.attendance_df) == 0:
            return pd.DataFrame(columns=[ "Name", "Total Hours", "Expected Hours", "LOH" ])

        # Group by name and date, sum duration
        attendance_by_day = self.attendance_df.groupby([ 'Name', 'Date' ])[ 'Duration' ].sum().reset_index()

        # Group by name to get total hours
        total_hours = attendance_by_day.groupby('Name')[ 'Duration' ].sum().reset_index()
        total_hours[ 'Duration' ] = total_hours[ 'Duration' ] / 60  # Convert minutes to hours

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
        """
        Display a window showing the LOH report
        """
        loh_data = self.calculate_loh()

        if len(loh_data) == 0:
            messagebox.showinfo("No Data", "No attendance records found.")
            return

        # Create a new window
        report_window = tk.Toplevel(self.root)
        report_window.title("Loss of Hours (LOH) Report")
        report_window.geometry("600x400")

        # Create a frame for the table
        frame = tk.Frame(report_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a treeview (table)
        tree = ttk.Treeview(frame)
        tree[ "columns" ] = list(loh_data.columns)
        tree[ "show" ] = "headings"

        # Define column headings
        for col in loh_data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)

        # Add data to the table
        for _, row in loh_data.iterrows():
            values = [ row[ col ] if col != 'Name' else row[ col ] for col in loh_data.columns ]
            # Format numerical values to 2 decimal places
            values = [ val if isinstance(val, str) else f"{val:.2f}" for val in values ]
            tree.insert("", tk.END, values=values)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)

        # Add export button
        export_button = tk.Button(report_window, text="Export Report",
                                  command=lambda: self.export_loh_report(loh_data),
                                  bg="#4CAF50", fg="white", padx=10, pady=5)
        export_button.pack(pady=10)

    def export_loh_report(self, loh_data):
        """
        Export the LOH report to a CSV file
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[ ("CSV files", "*.csv"), ("All files", "*.*") ],
            title="Save LOH Report"
        )

        if file_path:
            loh_data.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", f"Report exported to {file_path}")

    def set_expected_hours(self):
        """
        Open a dialog to set the expected hours per day
        """

        def save_hours():
            try:
                hours = float(hours_entry.get())
                if hours <= 0:
                    messagebox.showerror("Invalid Input", "Expected hours must be greater than 0")
                    return
                self.expected_hours = hours
                dialog.destroy()
                messagebox.showinfo("Success", f"Expected hours set to {hours} per day")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number")

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Expected Hours")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        # Add widgets
        tk.Label(dialog, text="Expected hours per day:").pack(pady=10)

        hours_entry = tk.Entry(dialog)
        hours_entry.insert(0, str(self.expected_hours))
        hours_entry.pack(pady=10)

        tk.Button(dialog, text="Save", command=save_hours,
                  bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=10)

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()