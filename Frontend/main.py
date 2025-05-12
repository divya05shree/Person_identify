import tkinter as tk
import cv2
from PIL import Image, ImageTk
import threading
import time
import queue
from Person_identify.faceAndPerson.FaceWithperson import FaceRecognition, FaceWithPerson
from Person_identify.Attendance.Attendence import attendence
from ultralytics import YOLO
import numpy as np


class VideoRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Recognition System")
        self.root.geometry("800x500")
        self.root.config(bg="#696969")  # Dark gray background
        self.track_person = YOLO('yolov8n.pt')

        # Variables
        self.recognizing = False
        self.cap = None
        self.detected = {}
        self.focused_name = ""
        self.is_focusing = False
        self.frame_queue = queue.Queue(maxsize=5)  # Frame buffer queue
        self.processed_frame_queue = queue.Queue(maxsize=5)  # Processed frames queue
        self.zoomed_frame_queue = queue.Queue(maxsize=5)  # Zoomed frames queue
        self.processing_active = False
        self.skip_frames = 2  # Process every n frames for detection
        self.frame_count = 0
        self.zoom_box = None  # Store the bounding box for the focused person

        # Create frames
        self.control_frame = tk.Frame(root, bg="#696969", width=300, height=500)
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        self.control_frame.pack_propagate(False)

        self.video_frame = tk.Frame(root, bg="#696969", width=500, height=500)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Control section elements
        self.create_control_elements()

        # Video section elements
        self.create_video_elements()

        # Initialize recognition components
        self.faceWithPerson = FaceWithPerson()
        self.faceRecognition = FaceRecognition()
        self.attendance = attendence()
        self.is_zoomed = False

        # Start video streams
        self.start_video_streams()

    def create_control_elements(self):
        # Recognition button
        self.recognition_btn = tk.Button(
            self.control_frame,
            text="Start recognizing",
            bg="black", fg="white",
            command=self.toggle_recognition,
            height=2, width=25
        )
        self.recognition_btn.pack(pady=10, padx=20)

        # Name input frame
        input_frame = tk.Frame(self.control_frame, bg="#696969")
        input_frame.pack(pady=5, fill=tk.X, padx=20)

        # Name input
        self.name_label = tk.Label(input_frame, text="Input to focus(Name)", bg="#696969", fg="white")
        self.name_label.pack(side=tk.LEFT, padx=5)

        self.name_entry = tk.Entry(input_frame, width=10)
        self.name_entry.pack(side=tk.LEFT, padx=5)

        self.focus_btn = tk.Button(
            input_frame,
            text="Start focus",
            bg="black", fg="white",
            command=self.toggle_focus
        )
        self.focus_btn.pack(side=tk.LEFT, padx=5)

        # WASD control buttons
        # W button
        self.w_btn = tk.Button(
            self.control_frame,
            text="W",
            bg="#ff6666", fg="black",
            width=5, height=2,
            command=lambda: self.move("up")
        )
        self.w_btn.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

        # A button
        self.a_btn = tk.Button(
            self.control_frame,
            text="A",
            bg="#ff6666", fg="black",
            width=5, height=2,
            command=lambda: self.move("left")
        )
        self.a_btn.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

        # S button
        self.s_btn = tk.Button(
            self.control_frame,
            text="S",
            bg="#ff6666", fg="black",
            width=5, height=2,
            command=lambda: self.move("down")
        )
        self.s_btn.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # D button
        self.d_btn = tk.Button(
            self.control_frame,
            text="D",
            bg="#ff6666", fg="black",
            width=5, height=2,
            command=lambda: self.move("right")
        )
        self.d_btn.place(relx=0.7, rely=0.5, anchor=tk.CENTER)

        # Print attendance button
        self.attendance_btn = tk.Button(
            self.control_frame,
            text="Print attendance",
            bg="#ff6666", fg="black",
            height=2,
            command=self.print_attendance
        )
        self.attendance_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

    def create_video_elements(self):
        # Upper video frame (without bounding box)
        self.video_frame1 = tk.Frame(self.video_frame, bg="#696969", width=500, height=250)
        self.video_frame1.pack(fill=tk.BOTH, expand=True)

        self.video_label1 = tk.Label(self.video_frame1, bg="#696969")
        self.video_label1.pack(fill=tk.BOTH, expand=True)

        self.video_text1 = tk.Label(self.video_frame1, text="Video without bounding box", bg="#696969", fg="#FFD700")
        self.video_text1.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Lower video frame (with bounding box)
        self.video_frame2 = tk.Frame(self.video_frame, bg="#696969", width=500, height=250)
        self.video_frame2.pack(fill=tk.BOTH, expand=True)

        self.video_label2 = tk.Label(self.video_frame2, bg="#696969")
        self.video_label2.pack(fill=tk.BOTH, expand=True)

        self.video_text2 = tk.Label(self.video_frame2, text="Video with bounding box", bg="#696969", fg="#FFD700")
        self.video_text2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def toggle_recognition(self):
        self.recognizing = not self.recognizing

        if self.recognizing:
            self.recognition_btn.config(text="Stop recognizing", bg="red")
            print("Recognition started")
            if not self.processing_active:
                self.processing_active = True
                self.processing_thread = threading.Thread(target=self.process_frames)
                self.processing_thread.daemon = True
                self.processing_thread.start()
        else:
            self.recognition_btn.config(text="Start recognizing", bg="black")
            print("Recognition stopped")
            self.processing_active = False

    def toggle_focus(self):
        name = self.name_entry.get()
        self.is_focusing = not self.is_focusing

        if self.is_focusing:
            if name:
                self.focused_name = name
                self.focus_btn.config(text="Stop focus", bg="red")
                print(f"Now focusing on: {name}")
                # Make sure recognition is on to detect faces
                if not self.recognizing:
                    self.toggle_recognition()
            else:
                print("Please enter a name to focus")
                self.is_focusing = False
        else:
            self.focused_name = ""
            self.focus_btn.config(text="Start focus", bg="black")
            print("Stopped focusing")
            self.zoom_box = None

    def move(self, direction):
        print(f"Moving camera: {direction}")
        # Here you would add logic to control camera movement

    def print_attendance(self):
        print("Printing attendance...")
        # Here you would add logic to print attendance record

    def start_video_streams(self):
        # Try to open camera
        try:
            self.cap = cv2.VideoCapture(0)  # Use default camera
            if not self.cap.isOpened():
                raise Exception("Could not open video device")

            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()

            # Start display thread
            self.display_thread = threading.Thread(target=self.display_frames)
            self.display_thread.daemon = True
            self.display_thread.start()

        except Exception as e:
            print(f"Error opening camera: {e}")
            # If camera isn't available, just show placeholder
            self.video_text1.config(text="Camera not available")
            self.video_text2.config(text="Camera not available")

    def capture_frames(self):
        """Continuously capture frames from camera and put them in queue"""
        latest_frame = None
        while True:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to RGB format
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                latest_frame = rgb_frame.copy()

                # Only put frame in queue if there's space (non-blocking)
                try:
                    self.frame_queue.put(rgb_frame, block=False)
                except queue.Full:
                    # If queue is full, remove oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(rgb_frame, block=False)
                    except:
                        pass

            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    def process_frames(self):
        """Process frames for person/face detection in a separate thread"""
        last_processed_frame = None

        while self.processing_active:
            try:
                # Get a frame to process
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(block=False)

                    # Skip frames to reduce processing load
                    self.frame_count += 1
                    if self.frame_count % self.skip_frames == 0:
                        # Process the frame with detection
                        processed_frame, zoom_info = self.find_user(frame)
                        last_processed_frame = processed_frame

                        # If we're focusing and found the person, create a zoomed frame
                        if self.is_focusing and zoom_info and zoom_info[ 'found' ]:
                            zoomed_frame = self.create_zoomed_frame(frame, zoom_info[ 'box' ])
                            # Put zoomed frame in queue
                            try:
                                self.zoomed_frame_queue.put(zoomed_frame, block=False)
                            except queue.Full:
                                try:
                                    self.zoomed_frame_queue.get_nowait()
                                    self.zoomed_frame_queue.put(zoomed_frame, block=False)
                                except:
                                    pass
                    else:
                        # If we're skipping this frame but have a previous processed frame,
                        # use that with current frame's dimensions
                        if last_processed_frame is not None:
                            processed_frame = last_processed_frame
                        else:
                            processed_frame = frame.copy()

                    # Put processed frame in queue (non-blocking)
                    try:
                        self.processed_frame_queue.put(processed_frame, block=False)
                    except queue.Full:
                        # If queue is full, remove oldest frame and add new one
                        try:
                            self.processed_frame_queue.get_nowait()
                            self.processed_frame_queue.put(processed_frame, block=False)
                        except:
                            pass

            except queue.Empty:
                pass

            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    def create_zoomed_frame(self, frame, box):
        """Create a zoomed frame focusing on the specified box"""
        x1, y1, x2, y2 = box

        # Add some padding around the person
        padding = 50
        h, w = frame.shape[ :2 ]

        # Ensure box coordinates stay within frame boundaries with padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Extract the region of interest
        zoomed_frame = frame[ y1:y2, x1:x2 ].copy()

        # Add a text label indicating zoom mode
        cv2.putText(
            zoomed_frame,
            f"Focusing on: {self.focused_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        return zoomed_frame

    def display_frames(self):
        """Update both video frames from queues"""
        last_regular_frame = None
        last_processed_frame = None
        last_zoomed_frame = None

        while True:
            # Get latest regular frame
            if not self.frame_queue.empty():
                try:
                    last_regular_frame = self.frame_queue.get(block=False)
                except queue.Empty:
                    pass

            # Get latest zoomed frame if available
            if self.is_focusing and not self.zoomed_frame_queue.empty():
                try:
                    last_zoomed_frame = self.zoomed_frame_queue.get(block=False)
                except queue.Empty:
                    pass

            # Display regular frame or zoomed frame if focusing
            if self.is_focusing and last_zoomed_frame is not None:
                self.display_frame(last_zoomed_frame, self.video_label1)
                # Display "Focusing on [Name]" text
                self.video_text1.config(text=f"Focusing on: {self.focused_name}")
            elif last_regular_frame is not None:
                self.display_frame(last_regular_frame, self.video_label1)
                # Reset text if not focusing
                if not self.is_focusing:
                    self.video_text1.config(text="Video without bounding box")

            # Get latest processed frame if recognition is on
            if self.recognizing and not self.processed_frame_queue.empty():
                try:
                    last_processed_frame = self.processed_frame_queue.get(block=False)
                except queue.Empty:
                    pass

            # Always display processed frame in bottom view if recognition is on
            if self.recognizing and last_processed_frame is not None:
                self.display_frame(last_processed_frame, self.video_label2)
            elif last_regular_frame is not None:
                # If recognition is off, just show the regular frame in both displays
                self.display_frame(last_regular_frame, self.video_label2)

            # Update the GUI
            self.root.update_idletasks()
            time.sleep(0.03)  # ~30 fps for display

    def display_frame(self, frame, label):
        """Convert a frame to PhotoImage and display on the given label"""
        h, w = frame.shape[ :2 ]
        # Calculate new dimensions to fit in the label while maintaining aspect ratio
        label_width = label.winfo_width()
        label_height = label.winfo_height()

        # If the label size hasn't been determined yet, set default
        if label_width < 10 or label_height < 10:
            label_width = 400
            label_height = 200

        # Calculate new dimensions
        ratio = min(label_width / w, label_height / h)
        new_width, new_height = int(w * ratio), int(h * ratio)

        # Resize the frame
        if new_width > 0 and new_height > 0:  # Avoid invalid dimensions
            resized = cv2.resize(frame, (new_width, new_height))
            img = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(image=img)

            # We need to keep a reference to the photo to prevent garbage collection
            label.imgtk = photo
            label.config(image=photo)

    def on_closing(self):
        """Clean up resources before closing"""
        self.processing_active = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def find_user(self, frame):
        processed_frame = frame.copy()  # Make a copy to avoid modifying original
        zoom_info = {'found': False, 'box': None}

        try:
            results = self.track_person.track(frame, verbose=False)  # Set verbose=False to reduce console output

            for result in results:
                for box in result.boxes:
                    # Get bounding box coordinates (xyxy format)
                    mask = np.zeros(frame.shape[ :2 ], dtype=np.uint8)
                    x1, y1, x2, y2 = map(int, box.xyxy[ 0 ])

                    # Check if track_id exists
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = box.id.int().cpu().tolist()
                        id_text = f"ID: {track_id[ 0 ]}"
                    else:
                        id_text = "ID: N/A"

                    # Draw filled rectangle for each detected person
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
                    cv2.putText(processed_frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),
                                2)

                    # Apply mask to frame
                    person = cv2.bitwise_and(frame, frame, mask=mask)

                    # Detect and recognize face
                    detect_face = self.faceWithPerson.detect_face(person)
                    identity = None

                    if detect_face:
                        x, y, w, h = detect_face
                        face_crop = person[ y:y + h, x:x + w ]
                        identity = self.faceRecognition.recognize_face(face_crop)

                    if identity:
                        self.attendance.take_attendence(identity)
                        cv2.putText(processed_frame, identity, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (36, 255, 12), 2)

                        # Check if this is the person we're focusing on
                        if self.is_focusing and identity == self.focused_name:
                            # Highlight the focused person with a different color
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for focused person
                            cv2.putText(processed_frame, "FOCUSED", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 0, 255), 2)

                            # Save the box for zooming
                            zoom_info = {'found': True, 'box': (x1, y1, x2, y2)}
                    else:
                        cv2.putText(processed_frame, "Unknown person", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2)

                    if identity and hasattr(box, 'id') and box.id is not None:
                        self.detected[ identity ] = track_id[ 0 ]

        except Exception as e:
            print(f"Error in detection: {e}")

        return processed_frame, zoom_info


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()