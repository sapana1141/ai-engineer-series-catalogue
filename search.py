import re
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Notice we removed ChromaDB from our config imports!
from config import DB_PATH, EMBED_MODEL

# Load the AI model
model = SentenceTransformer(EMBED_MODEL)

def tokenize(text):
    """Helper function to extract words from text for keyword matching."""
    return set(re.findall(r"\b\w+\b", text.lower()))

def cosine_similarity(v1, v2):
    """Calculates how similar two vectors are"""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search(query, k=10, filters=None):
    query = query.strip()
    if not query:
        return []

    # 1. Convert the user's query into a vector
    query_embedding = model.encode(query)

    # 2. Read ALL records and embeddings directly from SQLite
    conn = sqlite3.connect(DB_PATH)
    
    # We select 'embedding' which is stored as a BLOB (binary data)
    sql = """
        SELECT
            identifier, title, category, subcategory, 
            frequency, unit, search_text, embedding
        FROM series_catalogue
        WHERE 1=1
    """
    params = []

    # Apply any SQL filters if the user provided them
    if filters:
        for field, value in filters.items():
            allowed = {"category", "subcategory", "frequency", "unit", "currency", "discontinued"}
            field = field.lower()
            if field in allowed:
                sql += f" AND LOWER({field}) = LOWER(?)"
                params.append(str(value))

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    query_tokens = tokenize(query)
    ranked = []

    # 3. Score every row using Hybrid Search
    for row in rows:
        (identifier, title, category, subcategory, frequency, unit, search_text, embedding_blob) = row

        # Calculate Semantic Score (Vector math replacing ChromaDB)
        if embedding_blob:
            # Convert the binary BLOB back into a list of numbers
            record_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            # A perfect match is 1.0, worst is -1.0. We map it to a positive score.
            semantic_score = max(0, cosine_similarity(query_embedding, record_embedding))
        else:
            semantic_score = 0

        # Calculate Lexical Score (Keywords)
        text_tokens = tokenize(search_text or "")
        overlap = len(query_tokens & text_tokens)
        lexical_score = (overlap / len(query_tokens) if query_tokens else 0)

        # Calculate Title Score (Bonus for title match)
        title_tokens = tokenize(title or "")
        title_overlap = len(query_tokens & title_tokens)
        title_score = (title_overlap / len(query_tokens) if query_tokens else 0)

        # Final Hybrid Score
        final_score = (
            0.50 * lexical_score +
            0.30 * title_score +
            0.20 * semantic_score
        )

        if final_score > 0:
            ranked.append((final_score, identifier, title, category, subcategory, frequency, unit))

    # 4. Sort and return the best results
    ranked.sort(key=lambda x: (-x[0], x[1]))

    # Prune bad results if the top one is a near-perfect match
    if ranked and ranked[0][0] >= 0.90:
        top_score = ranked[0][0]
        ranked = [r for r in ranked if r[0] >= top_score * 0.92]

    return [
        {
            "identifier": row[1],
            "title": row[2],
            "category": row[3],
            "subcategory": row[4],
            "frequency": row[5],
            "unit": row[6],
            "score": round(row[0], 4)
        }
        for row in ranked[:k]
    ]

if __name__ == "__main__":
    print("Agent Search Engine Ready! (Powered by SQLite)")
    while True:
        user_query = input("\nSearch: ").strip()
        if not user_query:
            break
        
        results = search(user_query)
        for result in results:
            print(result)
