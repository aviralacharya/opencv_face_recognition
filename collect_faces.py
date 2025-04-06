# collect_faces.py
import cv2
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python collect_faces.py <name>")
    sys.exit(1)

name = sys.argv[1].strip()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

save_path = os.path.join("faces", name)
os.makedirs(save_path, exist_ok=True)

existing_files = [f for f in os.listdir(save_path) if f.endswith(".png")]
existing_indices = [int(f.split(".")[0]) for f in existing_files if f.split(".")[0].isdigit()]
start_index = max(existing_indices) + 1 if existing_indices else 0

cap = cv2.VideoCapture(0)
count = 0
target_count = 1000

while count < target_count:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (100, 100))

        img_index = start_index + count
        cv2.imwrite(os.path.join(save_path, f"{img_index}.png"), resized_face)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.putText(frame, f"Collected: {count}/{target_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
