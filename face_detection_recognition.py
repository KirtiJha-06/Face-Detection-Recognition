import cv2
import numpy as np
import os

# Create directories if not exist
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")

# Load the DNN face detection model
prototxt_path = cv2.data.haarcascades + "deploy.prototxt"
model_path = cv2.data.haarcascades + "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load pre-trained face data if exists
if os.path.exists("face_model.yml"):
    recognizer.read("face_model.yml")
    print("Face recognition model loaded!")

# Start video capture
cap = cv2.VideoCapture(0)

face_data = []  # Store face data for training
labels = []  # Store labels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            # Extract the face region
            face = frame[y:y1, x:x1]
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (100, 100))

            # Save the detected face for training
            face_filename = f"saved_faces/face_{len(face_data)}.jpg"
            cv2.imwrite(face_filename, resized_face)
            face_data.append(resized_face)
            labels.append(len(face_data))  # Assign a unique label

            # Recognize the face if the model is trained
            try:
                label, confidence = recognizer.predict(resized_face)
                if confidence < 50:  # Confidence threshold
                    cv2.putText(frame, f"Person {label}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except:
                cv2.putText(frame, "Training Required", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Face Detection & Recognition", frame)

    # Press 'q' to exit and train the model
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Train the recognizer
if len(face_data) > 5:  # Train only if there are enough images
    recognizer.train(face_data, np.array(labels))
    recognizer.save("face_model.yml")
    print("Face recognition model trained and saved!")

cap.release()
cv2.destroyAllWindows()
