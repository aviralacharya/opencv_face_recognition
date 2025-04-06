import tkinter as tk
from tkinter import messagebox
import subprocess
import threading

def run_collect_faces():
    def collect():
        name = name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter a name.")
            return
        subprocess.run(["python", "collect_faces.py", name], text=True)
    threading.Thread(target=collect).start()


def run_train_model():
    def train():
        subprocess.run(["python", "train_model.py"], text=True)
        messagebox.showinfo("Training Complete", "Model has been trained successfully.")
    threading.Thread(target=train).start()

def run_recognize():
    def recognize():
        subprocess.run(["python", "recognize.py"], text=True)
    threading.Thread(target=recognize).start()

# Create GUI
app = tk.Tk()
app.title("Face Recognition System")
app.geometry("400x300")
app.resizable(True, True)

# Title
tk.Label(app, text="Face Recognition System", font=("Helvetica", 16, "bold")).pack(pady=15)

# Name input
tk.Label(app, text="Enter Name:").pack()
name_entry = tk.Entry(app, font=("Helvetica", 12))
name_entry.pack(pady=5)

# Buttons
tk.Button(app, text="Collect Faces", command=run_collect_faces, width=25, height=2).pack(pady=10)
tk.Button(app, text="Train Model", command=run_train_model, width=25, height=2).pack(pady=10)
tk.Button(app, text="Recognize Faces", command=run_recognize, width=25, height=2).pack(pady=10)

app.mainloop()
