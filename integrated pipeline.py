from flask import Flask, request, jsonify
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import sqlite3
import os

app = Flask(__name__)

# Load FaceNet model
model = load_model("facenet_keras.h5")

# Load NLP model for text similarity
nlp = pipeline("feature-extraction", model="bert-base-uncased")

# Connect to SQLite database
conn = sqlite3.connect("dating_app.db", check_same_thread=False)
cursor = conn.cursor()

# Create users table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_embedding BLOB,
    description TEXT
)
''')
conn.commit()

# Step 1: Detect and align face
def detect_and_align_face(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    if not faces:
        raise ValueError("No face detected in the image.")
    x, y, width, height = faces[0]['box']
    face = image[y:y+height, x:x+width]
    face = cv2.resize(face, (160, 160))
    return face

# Step 2: Generate face embedding
def generate_face_embedding(face_image):
    face_image = face_image.astype('float32')
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std
    face_image = np.expand_dims(face_image, axis=0)
    embedding = model.predict(face_image)
    return embedding[0]

# Step 3: Store user data in database
def store_user_data(face_embedding, description):
    embedding_bytes = face_embedding.tobytes()
    cursor.execute("INSERT INTO users (face_embedding, description) VALUES (?, ?)",
                   (embedding_bytes, description))
    conn.commit()

# Step 4: Find matches based on facial similarity
def find_matches(user_embedding):
    cursor.execute("SELECT face_embedding, description FROM users")
    rows = cursor.fetchall()
    matches = []
    for row in rows:
        db_embedding = np.frombuffer(row[0], dtype=np.float32)
        similarity = cosine_similarity([user_embedding], [db_embedding])[0][0]
        matches.append((similarity, db_embedding, row[1]))  # Include description
    matches.sort(reverse=True, key=lambda x: x[0])
    return matches

# Step 5: Analyze description similarity
def calculate_text_similarity(text1, text2):
    embedding1 = np.mean(nlp(text1), axis=1)
    embedding2 = np.mean(nlp(text2), axis=1)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

# API endpoint to handle user request
@app.route("/match", methods=["POST"])
def match_user():
    # Get user data from request
    image_file = request.files["image"]
    description = request.form["description"]
    
    # Save the uploaded image temporarily
    image_path = "temp_user_photo.jpg"
    image_file.save(image_path)
    
    try:
        # Step 1: Detect and align face
        face_image = detect_and_align_face(image_path)
        
        # Step 2: Generate face embedding
        face_embedding = generate_face_embedding(face_image)
        
        # Step 3: Store user data
        store_user_data(face_embedding, description)
        
        # Step 4: Find matches
        matches = find_matches(face_embedding)
        
        # Step 5: Analyze description similarity
        results = []
        for match in matches[:5]:  # Top 5 matches
            db_description = match[2]
            text_similarity = calculate_text_similarity(description, db_description)
            results.append({
                "face_similarity": float(match[0]),
                "description_similarity": float(text_similarity),
                "description": db_description
            })
        
        # Clean up temporary file
        os.remove(image_path)
        
        return jsonify({"matches": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
