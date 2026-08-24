from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
import threading
import mediapipe as mp
import numpy as np
import pygame

app = Flask(__name__)

# Initialize pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")
alarm_playing = False
alarm_stopped = False
drowsy_triggered = False
alarm_lock = threading.Lock()

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# EAR threshold and duration
EAR_THRESHOLD = 0.25
CLOSED_EYES_DURATION = 2.0

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Shared status
status_text = "System idle"
current_camera = 0  # 0 for integrated, 1 for external
camera_lock = threading.Lock()
frame_buffer = None
frame_lock = threading.Lock()

# Pre-initialize both cameras
cap_integrated = None
cap_external = None
cameras_initialized = False

def init_cameras():
    global cap_integrated, cap_external, cameras_initialized
    try:
        # Initialize integrated camera (index 0)
        cap_integrated = cv2.VideoCapture(0)
        if cap_integrated.isOpened():
            cap_integrated.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap_integrated.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap_integrated.set(cv2.CAP_PROP_FPS, 30)
            print("✓ Integrated camera initialized")
        else:
            print("✗ Integrated camera not found")
        
        # Initialize external camera (index 1)
        cap_external = cv2.VideoCapture(1)
        if cap_external.isOpened():
            cap_external.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap_external.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap_external.set(cv2.CAP_PROP_FPS, 30)
            print("✓ External camera initialized")
        else:
            print("✗ External camera not found")
        
        cameras_initialized = True
    except Exception as e:
        print(f"Error initializing cameras: {e}")

def calculate_EAR(landmarks, eye_indices):
    eye = [landmarks[i] for i in eye_indices]
    A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
    B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
    C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
    ear = (A + B) / (2.0 * C)
    return ear

def play_alarm():
    global alarm_playing, drowsy_triggered
    with alarm_lock:
        if not alarm_playing:
            try:
                alarm_sound.play(-1)
                alarm_playing = True
                drowsy_triggered = True
                print("🔔 ALARM TRIGGERED!")
            except:
                print("Error playing alarm")

def stop_alarm():
    global alarm_playing, alarm_stopped, status_text, drowsy_triggered
    with alarm_lock:
        try:
            alarm_sound.stop()
        except:
            pass
        alarm_playing = False
        alarm_stopped = True
        drowsy_triggered = False
        status_text = "Alarm stopped"
        print("Alarm stopped")

def set_camera(camera_id):
    global current_camera
    with camera_lock:
        if current_camera != camera_id:
            current_camera = camera_id
            print(f"Switched to {'Integrated' if camera_id == 0 else 'External'} camera")
            return True
    return False

def get_camera():
    with camera_lock:
        return current_camera

def generate_frames():
    global alarm_stopped, status_text, drowsy_triggered, frame_buffer
    last_closed_time = None
    frame_count = 0
    
    while True:
        start_time = time.time()
        
        # Get current camera
        current_cam = get_camera()
        
        # Get frame from appropriate camera
        frame = None
        if current_cam == 0 and cap_integrated and cap_integrated.isOpened():
            ret, frame = cap_integrated.read()
        elif current_cam == 1 and cap_external and cap_external.isOpened():
            ret, frame = cap_external.read()
        
        if not ret or frame is None:
            # Show error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Camera {current_cam} not available!", 
                       (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        
        # Process every 2nd frame for better performance
        frame_count += 1
        process_face = (frame_count % 2 == 0)
        
        if process_face:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
        else:
            results = None

        # Status logic
        if drowsy_triggered:
            status_text = "⚠️ DROWSY! ALARM ACTIVE ⚠️"
            alarm_stopped = False
        elif alarm_stopped:
            status_text = "🔕 Alarm stopped"
        else:
            status_text = "✅ Monitoring..."

        # Drowsiness detection
        ear_value = 0
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_EAR = calculate_EAR(face_landmarks.landmark, LEFT_EYE)
                right_EAR = calculate_EAR(face_landmarks.landmark, RIGHT_EYE)
                ear_value = (left_EAR + right_EAR) / 2.0

                if ear_value < EAR_THRESHOLD:
                    if last_closed_time is None:
                        last_closed_time = time.time()
                    elif time.time() - last_closed_time > CLOSED_EYES_DURATION:
                        play_alarm()
                else:
                    last_closed_time = None

        # Set color based on status
        if "DROWSY" in status_text:
            color = (0, 0, 255)  # Red
        elif "Monitoring" in status_text:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 255, 255)  # Yellow
        
        # Display status
        cv2.putText(frame, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display camera info with highlight
        if current_cam == 0:
            camera_name = "📷 INTEGRATED CAMERA (ACTIVE)"
            cv2.rectangle(frame, (10, 50), (350, 90), (0, 255, 0), 2)
        else:
            camera_name = "🔌 EXTERNAL CAMERA (ACTIVE)"
            cv2.rectangle(frame, (10, 50), (350, 90), (255, 0, 0), 2)
        
        cv2.putText(frame, camera_name, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display EAR value if face detected
        if results and results.multi_face_landmarks:
            cv2.putText(frame, f"Eye Aspect Ratio: {ear_value:.2f}", (30, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add FPS counter
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to control frame rate
        time.sleep(0.01)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm_route():
    stop_alarm()
    return jsonify({'status': 'success'})

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    data = request.get_json()
    camera_type = data.get('camera', 'integrated')
    
    if camera_type == 'external':
        if cap_external and cap_external.isOpened():
            set_camera(1)
            return jsonify({'status': 'success', 'camera': 'external'})
        else:
            return jsonify({'status': 'error', 'message': 'External camera not available'}), 400
    else:
        if cap_integrated and cap_integrated.isOpened():
            set_camera(0)
            return jsonify({'status': 'success', 'camera': 'integrated'})
        else:
            return jsonify({'status': 'error', 'message': 'Integrated camera not available'}), 400

@app.route('/check_cameras', methods=['GET'])
def check_cameras():
    """Check if both cameras are available"""
    integrated_available = cap_integrated is not None and cap_integrated.isOpened()
    external_available = cap_external is not None and cap_external.isOpened()
    
    return jsonify({
        'integrated': integrated_available,
        'external': external_available,
        'current': get_camera()
    })

@app.route('/reconnect_cameras', methods=['POST'])
def reconnect_cameras():
    """Reconnect cameras if needed"""
    global cap_integrated, cap_external
    try:
        if cap_integrated:
            cap_integrated.release()
        if cap_external:
            cap_external.release()
        
        cap_integrated = cv2.VideoCapture(0)
        cap_external = cv2.VideoCapture(1)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Initialize both cameras at startup
    init_cameras()
    
    # Print status
    print("\n" + "="*50)
    print("CAMERA STATUS:")
    print(f"Integrated Camera (Index 0): {'✓ AVAILABLE' if cap_integrated and cap_integrated.isOpened() else '✗ NOT AVAILABLE'}")
    print(f"External Camera (Index 1): {'✓ AVAILABLE' if cap_external and cap_external.isOpened() else '✗ NOT AVAILABLE'}")
    print("="*50 + "\n")
    
    if not (cap_integrated and cap_integrated.isOpened()) and not (cap_external and cap_external.isOpened()):
        print("⚠️ WARNING: No cameras detected! Please check your camera connections.")
    
    app.run(debug=True, threaded=True)