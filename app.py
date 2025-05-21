from flask import Flask, Response, render_template
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from threading import Thread


model = load_model('mask_detector_model.keras')
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

app = Flask(__name__)

camera_active = False
cap = None

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype("float") / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Video stream generator
def generate_frames():
    global cap, camera_active
    cap = cv2.VideoCapture(0)
    while camera_active:
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

                if prediction > 0.5:
                    label = 'No Mask'
                    color = (0, 0, 255)  
                else:
                    label = 'Mask'
                    color = (0, 255, 0)  

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera_active
    if camera_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return '', 204  

@app.route('/toggle_camera')
def toggle_camera():
    global camera_active
    camera_active = not camera_active  
    if camera_active:
        # Run the video stream in a separate thread to avoid blocking the main thread
        Thread(target=generate_frames).start()
    return '', 204  

if __name__ == "__main__":
    app.run(debug=True)
