from tensorflow.keras.models import load_model
import numpy as np

# Load a pre-trained FaceNet model (you can download one online)
model = load_model("facenet_keras.h5")

def generate_face_embedding(face_image):
    # Preprocess the image for FaceNet
    face_image = face_image.astype('float32')
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std
    face_image = np.expand_dims(face_image, axis=0)
    
    # Generate embedding
    embedding = model.predict(face_image)
    return embedding[0]

# Example usage
face_embedding = generate_face_embedding(face_image)
print("Face Embedding:", face_embedding)
