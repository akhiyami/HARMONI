import numpy as np
import json

import heapq

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import NUM_TOP_FEATURES
from conversation.memory.models import ContextualFeature


COSINE_THRESHOLD = 0.2

# rag_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# rag_model = SentenceTransformer("google/embeddinggemma-300m")
rag_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def attach_embeddings(feature: ContextualFeature) -> ContextualFeature:
    if not feature.embeddings:
        feature.embeddings = [
            rag_model.encode(val, normalize_embeddings=True).tolist()
            for val in feature.value
        ]
    return feature

def coefs(N, r):
    """
    Generate N decreasing numbers whose sum is 1.
    
    Parameters:
        N (int): Number of numbers
        r (float): Decreasing ratio, 0 < r < 1
    
    Returns:
        list of float: Decreasing numbers summing to 1
    """
    if not (0 < r < 1):
        raise ValueError("r must be between 0 and 1")
    
    a = (1 - r) / (1 - r**N)  # first term
    numbers = [a * r**i for i in range(N)]
    return numbers

def features_retriever(question, conn, user_id, last_interaction=None) -> list:  
    """
    Retrieve relevant features from the memory based on the question.
    Returns a JSON-like list of dicts (memory objects).
    """
    memory = []
    cursor = conn.cursor()

    # -------- Primary Features (always included) -------- #
    cursor.execute(f"SELECT name, description, value FROM {user_id} WHERE type = 'primary'")
    rows = cursor.fetchall()
    for name, description, value in rows:
        feature = {
            "type": "primary",
            "name": name,
            "description": description,
            "value": value.split(";") if value else []
        }
        memory.append(feature)

    # -------- Contextual Features (similarity search) -------- #
    # Encode the query once
    q_emb = rag_model.encode(question, normalize_embeddings=True)


    past_embeddings = []
    past_weights = []
    if last_interaction:
        for interaction in last_interaction:
            past_embeddings.append(rag_model.encode(interaction['content'], normalize_embeddings=True))
        past_weights = coefs(len(past_embeddings), 0.5)

    # Fetch contextual features (with embeddings if stored)
    cursor.execute(f"SELECT rowid, name, description, value, embeddings FROM {user_id} WHERE type = 'contextual'")
    rows = cursor.fetchall()

    results = []  # (similarity, rowid, value_idx)

    for rowid, name, description, value, embeddings_blob in rows:
        if not value:
            continue

        values = value.split(";") if value else []

        if not embeddings_blob:
            #update the embeddings in the database
            feature = ContextualFeature(
                type="contextual",
                name=name,
                description=description,
                value=values or [],
                embeddings=None
            )
            feature = attach_embeddings(feature)
            cursor.execute(
                f"UPDATE {user_id} SET embeddings = ? WHERE rowid = ?",
                (json.dumps(feature.embeddings), rowid)
            )
            conn.commit()
            embeddings = feature.embeddings
        else:
            # Load embeddings back from JSON
            embeddings = json.loads(embeddings_blob)

        # Compare query with each value embedding
        for idx, emb in enumerate(embeddings):
            emb = np.array(emb, dtype=np.float32)  # convert back to numpy
            try:
                sim = cosine_similarity(
                    np.array(q_emb).reshape(1, -1),
                    emb.reshape(1, -1)
                ).flatten()[0]
            except Exception as e:
                continue

            if len(past_embeddings) > 0:
                past_sims = 0
                for past_emb, w in zip(past_embeddings, past_weights):
                    past_sim = cosine_similarity(
                        np.array(past_emb).reshape(1, -1),
                        emb.reshape(1, -1)
                    ).flatten()[0]
                    past_sims += past_sim * w
            
                DECAY_FACTOR = 0.7
                sim = DECAY_FACTOR * sim + (1 - DECAY_FACTOR) * past_sims

            if sim > COSINE_THRESHOLD:
                results.append((sim, rowid, idx))
            
    # -------- Select Top-N -------- #
    topN = heapq.nlargest(NUM_TOP_FEATURES, results, key=lambda x: x[0])

    for _, rowid, idx in topN:
        cursor.execute(f"SELECT name, description, value FROM {user_id} WHERE rowid = ?", (rowid,))
        row = cursor.fetchone()
        if row:
            name, description, value = row
            values = value.split(";")
            feature = {
                "type": "contextual",
                "name": name,
                "description": description,
                "value": [values[idx]] if values else []
            }
            memory.append(feature)

    return memory