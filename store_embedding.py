import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("dating_app.db")
cursor = conn.cursor()

# Create a table for user embeddings
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_embedding BLOB,
    description TEXT
)
''')
conn.commit()

def store_user_data(face_embedding, description):
    # Convert embedding to bytes for storage
    embedding_bytes = face_embedding.tobytes()
    
    # Insert into database
    cursor.execute("INSERT INTO users (face_embedding, description) VALUES (?, ?)",
                   (embedding_bytes, description))
    conn.commit()

# Example usage
store_user_data(face_embedding, "Loves hiking and photography.")
