import cv2
import tkinter as tk
from tkinter import messagebox
from deepface import DeepFace
from PIL import Image, ImageTk

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# إعداد نافذة Tkinter
root = tk.Tk()
root.title("Emotion Detector")

# إعداد عنصر لعرض الفيديو في الواجهة
label = tk.Label(root)
label.pack()

# دالة لعرض الفيديو في واجهة Tkinter
def show_frame():
    ret, frame = cap.read()
    if not ret:
        print("فشل في فتح الكاميرا.")
        return

    # تحويل الصورة من OpenCV إلى صورة يمكن عرضها باستخدام Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    
    label.imgtk = imgtk
    label.configure(image=imgtk)
    
    label.after(10, show_frame)

# دالة لالتقاط الصورة وتحليل المشاعر
def capture_and_analyze():
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("خطأ", "فشل في التقاط الصورة.")
        return
    
    # حفظ الصورة
    cv2.imwrite("captured.jpg", frame)
    print("تم حفظ الصورة.")
    
    try:
        # تحليل المشاعر
        result = DeepFace.analyze(img_path="captured.jpg", actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]["dominant_emotion"]
        
        # عرض نتيجة المشاعر في مربع حوار
        messagebox.showinfo("تحليل المشاعر", f"المشاعر السائدة: {dominant_emotion}")
        
        # عرض الصورة مع اسم المشاعر
        img = cv2.imread("captured.jpg")
        cv2.putText(img, f"Emotion: {dominant_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        messagebox.showerror("فشل التحليل", f"حدث خطأ في التحليل: {str(e)}")

# إضافة زر لالتقاط الصورة وتحليل المشاعر مع تخصيص مظهره
button = tk.Button(root, 
                   text="التقاط الصورة وتحليل المشاعر", 
                   command=capture_and_analyze,
                   font=("Arial", 14, "bold"),  # تخصيص الخط
                   bg="#4CAF50",  # اللون الخلفي
                   fg="white",  # اللون النص
                   relief="raised",  # شكل الزر
                   padx=20, pady=10,  # المسافة حول النص
                   bd=5,  # سمك الحواف
                   activebackground="#45a049",  # اللون عند التمرير على الزر
                   activeforeground="white")  # النص عند التمرير
button.pack(pady=20)

# بدء عرض الكاميرا
show_frame()

# تشغيل واجهة Tkinter
root.mainloop()

# إغلاق الكاميرا عند إنهاء البرنامج
cap.release()
cv2.destroyAllWindows()
