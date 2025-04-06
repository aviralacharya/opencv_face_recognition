# train_model.py
import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

faces = []
labels = []
label_map = {}
current_label = 0

image_size = (100, 100)

for person_name in os.listdir('faces'):
    person_folder = os.path.join('faces', person_name)
    if os.path.isdir(person_folder):
        label_map[current_label] = person_name
        for filename in os.listdir(person_folder):
            if filename.endswith(".png"):
                image_path = os.path.join(person_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                resized = cv2.resize(image, image_size)
                faces.append(resized.flatten())
                labels.append(current_label)
        current_label += 1

faces = np.array(faces)
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model and label map
joblib.dump((model, label_map), 'face_model_with_labels.pkl')

# Accuracy check
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
