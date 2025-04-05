import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare the dataset
faces = []
labels = []
label_map = {}
current_label = 0

# Define the target image size (e.g., 100x100)
image_size = (100, 100)

# Load the saved face data
for person_name in os.listdir('faces'):
    person_folder = os.path.join('faces', person_name)
    if os.path.isdir(person_folder):
        label_map[current_label] = person_name
        for filename in os.listdir(person_folder):
            if filename.endswith(".npy"):
                face_data = np.load(os.path.join(person_folder, filename))

                # Resize face data to a consistent size (e.g., 100x100)
                face_data_resized = cv2.resize(face_data, image_size)

                faces.append(face_data_resized)
                labels.append(current_label)
        current_label += 1

# Convert the list of faces and labels into numpy arrays
faces = np.array(faces)
labels = np.array(labels)



faces = faces.reshape(faces.shape[0], -1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Initialize and train the classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save the model
import joblib
joblib.dump(model, 'face_recognition_model.pkl')

# Test the model (optional)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
