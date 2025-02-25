from transformers import pipeline

# Load a pre-trained NLP model for text similarity
nlp = pipeline("feature-extraction", model="bert-base-uncased")

def calculate_text_similarity(text1, text2):
    # Generate embeddings for both texts
    embedding1 = np.mean(nlp(text1), axis=1)
    embedding2 = np.mean(nlp(text2), axis=1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

# Example usage
user_description = "Loves hiking and photography."
db_description = "Enjoys outdoor activities and taking pictures."
similarity_score = calculate_text_similarity(user_description, db_description)
print("Description Similarity:", similarity_score)
