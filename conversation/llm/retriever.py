import numpy as np

import heapq

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import NUM_TOP_FEATURES


COSINE_THESHOLD = 0.5

# rag_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
rag_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def features_retriever(question, conn, user_id):  
    """
    Retrieve relevant features from the memory based on the question.
    """
    memory = []

    # Add the primary features to the memory
    cursor = conn.cursor()
    cursor.execute(f"SELECT name, value FROM {user_id} WHERE type = 'primary'")
    rows = cursor.fetchall()
    for row in rows:
        name, value = row
        feature = {
            "type": "primary",
            "name": name,
            "value": value.split(";") if value else []
        }
        memory.append(feature)

    # Convert the question into embeddings
    question_embeddings = rag_model.encode(question, convert_to_tensor=True).cpu().numpy()  

    # Update embeddings for features that do not have them
    cursor.execute(f"SELECT rowid, value FROM {user_id} WHERE type = 'contextual'")
    rows = cursor.fetchall()

    results = []  # store (similarity, rowid, valid)
    for row in rows:
        rowid, value = row
        if not value:
            continue

        value_list = value.split(";")
        embeddings = np.array([rag_model.encode(value) for value in value_list], dtype=np.float32)

        for valid, embedding in enumerate(embeddings):
            cs = cosine_similarity(
                question_embeddings.reshape(1, -1),
                np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
            ).flatten()[0]

            if cs > COSINE_THESHOLD:  # keep only above threshold
                results.append((cs, rowid, valid))

    # Get top-N by similarity
    topN = heapq.nlargest(NUM_TOP_FEATURES, results, key=lambda x: x[0])

    # Extract (rowid, valid) pairs
    final_pairs = [(rowid, valid) for _, rowid, valid in topN]

    if final_pairs:
        for rowid, valid in final_pairs:
            cursor.execute(f"SELECT name, value FROM {user_id} WHERE rowid = ?", (rowid,))
            row = cursor.fetchone()
            if row:
                name, value = row
                feature = {
                    "type": "contextual",
                    "name": name,
                    "value": value.split(";")[valid] if value else []
                }
                memory.append(feature)

    return memory