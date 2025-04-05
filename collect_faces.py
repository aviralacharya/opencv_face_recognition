import cv2
import numpy as np
import os

# Initialize Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

name = input("Enter your name: ").strip()
save_path = os.path.join("faces", name)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while count < 5000:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Resize the face to a standard size (e.g., 100x100)
        resized_face = cv2.resize(gray_face, (100, 100))

        # Flatten the face to a 1D array and save it
        data = resized_face.flatten()
        np.save(os.path.join(save_path, f"{count}.npy"), data)

        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.putText(frame, f"Collected: {count}/50", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
