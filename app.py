import cv2
import tkinter as tk
from tkinter import messagebox
from deepface import DeepFace
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO


cap = cv2.VideoCapture(0)


model = YOLO("yolov8n.pt")


root = tk.Tk()
root.title("Face, Emotion, and Object Detection")


label = tk.Label(root)
label.pack()


current_frame = None


def show_frame():
    global current_frame
    ret, frame = cap.read()
    if not ret:
        print("فشل في فتح الكاميرا.")
        return

    current_frame = frame.copy()

   
    results = model(frame)
    detections = results[0].boxes.xywh.cpu().numpy()
    names = results[0].names

    for i, detection in enumerate(detections):
        x, y, w, h = detection[:4]
        conf = detection[4] if len(detection) > 4 else 0
        cls = int(results[0].boxes.cls[i]) if len(results[0].boxes.cls) > i else 0

        if conf > 0.5:
            obj_label = f"{names[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
            cv2.putText(frame, obj_label, (int(x - w / 2), int(y - h / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]["dominant_emotion"]
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print(f"فشل تحليل المشاعر: {str(e)}")

   
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, show_frame)


def capture_and_save(event=None):
    if current_frame is not None:
        filename = "snapshot.png"
        cv2.imwrite(filename, current_frame)
        messagebox.showinfo("تم الالتقاط", f"تم حفظ الصورة باسم {filename}")
    else:
        messagebox.showerror("خطأ", "لا توجد صورة حالية لحفظها.")


root.bind("<Return>", capture_and_save)  # Enter
root.bind("<space>", capture_and_save)   # Spacebar


button = tk.Button(root,
                   text="التقاط وحفظ الصورة (Enter أو Space)",
                   command=capture_and_save,
                   font=("Arial", 14, "bold"),
                   bg="#2196F3", fg="white",
                   relief="raised", padx=20, pady=10, bd=4,
                   activebackground="#1976D2", activeforeground="white")
button.pack(pady=15)


def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)


show_frame()
root.mainloop()
