import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained mask detection model
model = load_model('mask_detector_model.h5')

# Only two valid classes now!
labels = ['with_mask', 'without_mask']

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_mask(face_img):
    face = cv2.resize(face_img, (224, 224))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    preds = model(face, training=False).numpy()
    print(f"Raw predictions: {preds}")  # debug: see values for both classes!
    label_idx = np.argmax(preds)
    return labels[label_idx], preds[0][label_idx]

# Open webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        label, confidence = predict_mask(face_img)
        color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.imshow('Mask Detection - Press Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
