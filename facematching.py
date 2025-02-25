from sklearn.metrics.pairwise import cosine_similarity

def find_matches(user_embedding):
    # Fetch all embeddings from the database
    cursor.execute("SELECT face_embedding FROM users")
    rows = cursor.fetchall()
    
    matches = []
    for row in rows:
        db_embedding = np.frombuffer(row[0], dtype=np.float32)
        similarity = cosine_similarity([user_embedding], [db_embedding])[0][0]
        matches.append((similarity, db_embedding))
    
    # Sort matches by similarity
    matches.sort(reverse=True, key=lambda x: x[0])
    return matches

# Example usage
matches = find_matches(face_embedding)
print("Top Matches:", matches[:5])
