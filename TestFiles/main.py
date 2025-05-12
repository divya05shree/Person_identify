import tkinter as tk
from Person_identify.TestFiles.yolotest import VideoApp

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root, "Person Detection and Recognition System")