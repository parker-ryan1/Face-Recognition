from mtcnn import MTCNN
import cv2

def detect_and_align_face(image_path):
    # Load image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    # Detect faces
    detector = MTCNN()
    faces = detector.detect_faces(image)
    
    if not faces:
        raise ValueError("No face detected in the image.")
    
    # Extract the bounding box of the first face
    x, y, width, height = faces[0]['box']
    face = image[y:y+height, x:x+width]
    
    # Resize to a standard size (e.g., 160x160 for FaceNet)
    face = cv2.resize(face, (160, 160))
    
    return face

# Example usage
face_image = detect_and_align_face("user_photo.jpg")
cv2.imwrite("aligned_face.jpg", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
