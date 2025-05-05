import torch
import numpy as np
from PIL import Image
import cv2
import os
import pickle
from flask import Flask, request, render_template, flash, redirect, url_for, Response, send_from_directory
from torchvision import transforms
from werkzeug.utils import secure_filename

# Import Siamese Network
from model import SiameseNetwork

# Flask setup with explicit template and static folders
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['SECRET_KEY'] = 'zohaib_siamese'  # secure key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load('models/siamese_model.pth', map_location=device))
model.eval()

# Load face detector
face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Error loading Haar Cascade")

# Transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load or initialize embeddings database
EMBEDDINGS_FILE = 'embeddings.pkl'
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, 'rb') as f:
        known_embeddings = pickle.load(f) 
else:
    known_embeddings = {}

# Function to detect and crop face
def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
        return None, None
    
    # Take the first face
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Save face with bounding box
    output_filename = 'detected_' + os.path.basename(image_path)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    if not cv2.imwrite(output_path, img):
        print(f"Error: Failed to save {output_path}")
        return None, None
    print(f"Saved detected image: {output_path}")
    
    return face_rgb, output_path

# Function to get embedding
def get_embedding(image):
    if image is None:
        return None
    img = Image.fromarray(image)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(img) if hasattr(model, 'forward_once') else model(img, img)[0]
    return embedding.cpu().numpy()

# Function to compare faces
def compare_faces(img1_path, img2_path, threshold=0.5):
    face1, det1_path = detect_face(img1_path)
    face2, det2_path = detect_face(img2_path)
    
    if face1 is None or face2 is None or det1_path is None or det2_path is None:
        print(f"Face detection failed: img1_path={img1_path}, img2_path={img2_path}")
        return None, None, None, None, "No face detected or failed to save image"
    
    emb1 = get_embedding(face1)
    emb2 = get_embedding(face2)
    
    if emb1 is None or emb2 is None:
        print("Error processing embeddings")
        return None, None, None, None, "Error processing embeddings"
    
    distance = np.linalg.norm(emb1 - emb2)
    prediction = distance < threshold
    
    min_dist = float('inf')
    closest_name = "Unknown"
    for name, known_emb in known_embeddings.items():
        dist = np.linalg.norm(emb1 - known_emb)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            closest_name = name
    
    print(f"Compare faces: distance={distance:.4f}, prediction={'Same' if prediction else 'Different'}, closest_name={closest_name}")
    return det1_path, det2_path, distance, prediction, closest_name

# Function to process webcam frame
def process_webcam_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               cv2.imencode('.jpg', cv2.putText(np.zeros((480, 640, 3), dtype=np.uint8),
                                               "No webcam detected", (50, 240),
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2))[1].tobytes() +
               b'\r\n')
        return
    
    print("Webcam opened successfully")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        closest_name = "Unknown"
        min_dist = float('inf')
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            embedding = get_embedding(face_rgb)
            
            if embedding is not None:
                for name, known_emb in known_embeddings.items():
                    dist = np.linalg.norm(embedding - known_emb)
                    if dist < min_dist and dist < 0.5:
                        min_dist = dist
                        closest_name = name
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{closest_name} ({min_dist:.4f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    print("Webcam released")

# Debug route to serve images directly
@app.route('/debug/image/<path:filename>')
def debug_image(filename):
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Debug image requested: {full_path}, exists: {os.path.exists(full_path)}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for uploading images
@app.route('/', methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            flash('Please upload two images', 'error')
            return redirect(request.url)
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        if file1.filename == '' or file2.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file1 and file2:
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file1.save(path1)
            file2.save(path2)
            
            det1_path, det2_path, distance, prediction, closest_name = compare_faces(path1, path2)
            
            if distance is None:
                flash('Error: ' + closest_name, 'error')
                return redirect(request.url)
            
            # Convert to relative paths for url_for
            img1_rel = os.path.join('uploads', os.path.basename(det1_path)).replace('\\', '/') if det1_path and os.path.exists(det1_path) else 'placeholder.jpg'
            img2_rel = os.path.join('uploads', os.path.basename(det2_path)).replace('\\', '/') if det2_path and os.path.exists(det2_path) else 'placeholder.jpg'
            
            print(f"Rendering images - img1: {img1_rel}, img2: {img2_rel}")
            print(f"Image 1 exists: {os.path.exists(os.path.join('static', img1_rel))}")
            print(f"Image 2 exists: {os.path.exists(os.path.join('static', img2_rel))}")
            
            return render_template('result.html',
                                 img1=img1_rel,
                                 img2=img2_rel,
                                 distance=f"{distance:.4f}",
                                 prediction="Same person" if prediction else "Different people",
                                 closest_name=closest_name)
    
    return render_template('upload.html')

# Route for registering new person
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'file' not in request.files or 'name' not in request.form:
            flash('Please upload an image and provide a name', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        name = request.form['name'].strip()
        
        if file.filename == '' or not name:
            flash('No selected file or name', 'error')
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        face, _ = detect_face(path)
        if face is None:
            flash('No face detected in the image', 'error')
            return redirect(request.url)
        
        embedding = get_embedding(face)
        if embedding is None:
            flash('Error processing embedding', 'error')
            return redirect(request.url)
        
        known_embeddings[name] = embedding
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(known_embeddings, f)
        
        flash(f'Successfully registered {name}', 'success')
        return redirect(url_for('upload_images'))
    
    return render_template('upload.html', register=True)

# Route for webcam streaming
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Route for webcam feed
@app.route('/video_feed')
def video_feed():
    return Response(process_webcam_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, threaded=True)