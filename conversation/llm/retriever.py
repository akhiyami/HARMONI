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
            "description": None,
            "tags": None,
            "value": value.split(";") if value else []
        }
        memory.append(feature)

    # Convert the question into embeddings
    question_embeddings = rag_model.encode(question, convert_to_tensor=True).cpu().numpy()  

    #Update embeddings for features that do not have them
    cursor.execute(f"SELECT rowid, tags, embeddings FROM {user_id} WHERE type = 'contextual'")
    rows = cursor.fetchall()
    cosine_similarities = {}
    for row in rows:
        rowid, tags, embedding = row
        if embedding is None:
            if tags is None:
                tags = ""
            tags_list = tags.split(";")
            embedding = rag_model.encode(", ".join(tags_list)).tolist()
            embedding = np.array(embedding, dtype=np.float32).tobytes()  #Convert to bytes for BLOB storage
            cursor.execute(
                f"UPDATE {user_id} SET embeddings = ? WHERE rowid = ?",
                (embedding, rowid)
            )
            conn.commit()
        # If the embedding already exists, retrieve it
        else:
            embedding = np.frombuffer(embedding, dtype=np.float32)

        # Calculate cosine similarity between the question and the feature embeddings
        cs = cosine_similarity(
            question_embeddings.reshape(1, -1), 
            np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
        ).flatten()
        cosine_similarities[rowid] = cs[0]

        name_feature = cursor.execute(f"SELECT name FROM {user_id} WHERE rowid = ?", (rowid,)).fetchone()[0]
        # print(f"Feature '{name_feature}' (rowid {rowid}) has cosine similarity {cs[0]:.4f}")

        # Filter features based on a cosine similarity threshold
        filtered = [(rowid, sim) for rowid, sim in cosine_similarities.items() if sim > COSINE_THESHOLD]
        
        # Sort and limit the results to the top n most relevant features
        top_filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:NUM_TOP_FEATURES]
        top_indices = [rowid for rowid, _ in top_filtered]
        
        # Retrieve the top features from the database
        if top_indices:
            cursor.execute(
                f"SELECT name, description, tags, value FROM {user_id} WHERE rowid IN ({','.join(['?']*len(top_indices))})", 
                top_indices
            )
            rows = cursor.fetchall()
        else:
            rows = []

    # Format the retrieved data into a list of Feature objects
    for row in rows:
        name, description, tags, value = row
        feature = {
            "type": "contextual",  
            "name": name,
            "description": description,
            "tags": tags.split(";") if tags else [],
            "value": value.split(";")
        }
        memory.append(feature)

    return memory

def features_values_retriever(question, conn, user_id):  
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
            "description": None,
            "tags": None,
            "value": value.split(";") if value else []
        }
        memory.append(feature)

    # Convert the question into embeddings
    question_embeddings = rag_model.encode(question, convert_to_tensor=True).cpu().numpy()  

    #Update embeddings for features that do not have them
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
            cursor.execute(f"SELECT name, description, tags, value FROM {user_id} WHERE rowid = ?", (rowid,))
            row = cursor.fetchone()
            if row:
                name, description, tags, value = row
                feature = {
                    "type": "contextual",
                    "name": name,
                    "description": description,
                    "tags": tags.split(";") if tags else [],
                    "value": value.split(";")[valid] if value else []
                }
                memory.append(feature)

    return memory