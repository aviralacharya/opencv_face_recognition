# recognize.py
import cv2
import numpy as np
import joblib

# Load the trained model and label map
model, label_map = joblib.load('face_model_with_labels.pkl')

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Define distance threshold (tune as needed)
UNKNOWN_THRESHOLD = 5000.0  # You can tweak this value after testing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized = cv2.resize(face, (100, 100))
        flat = resized.flatten().reshape(1, -1)

        # Get the distance to nearest neighbor
        distances, indices = model.kneighbors(flat, n_neighbors=1)
        distance = distances[0][0]

        if distance < UNKNOWN_THRESHOLD:
            label = model.predict(flat)[0]
            name = label_map.get(label, "Unknown")
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
