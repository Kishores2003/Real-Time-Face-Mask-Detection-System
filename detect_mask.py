import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


model = load_model('mask_detector_model.keras')
print("Model loaded successfully")


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype("float") / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            face_img = preprocess_face(face_img)
            prediction = model.predict(face_img)[0][0]
            print(f"Face coordinates: {(x, y, x+w, y+h)}")
            print(f"Prediction value: {prediction}")

            if prediction > 0.5:
                label = 'No Mask'
                color = (0, 0, 255)  
                
            else:
                label = 'Mask'
                color = (0, 255, 0)  

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Mask Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
