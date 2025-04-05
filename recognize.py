import cv2
import numpy as np
import joblib

# Load the trained model
model = joblib.load('face_recognition_model.pkl')

# Define the label map to map the labels (indices) to the names of the people
label_map = {
    0: "Abiral",  # Corresponds to the first person in the training set
    1: "Ram",     # Second person
    2: "Boy",     # Third person
    3: "Asgya",   # Fourth person
    # Add more mappings based on your dataset
}

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture (camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the frame
        face = gray[y:y + h, x:x + w]
        
        # Resize the face to the size the model expects (e.g., 100x100)
        face_resized = cv2.resize(face, (100, 100))

        # Flatten the face image to match the model's input format
        face_flattened = face_resized.flatten().reshape(1, -1)

        # Predict the label
        label = model.predict(face_flattened)

        predicted_name = label_map.get(label[0], "Unknown")


        cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
