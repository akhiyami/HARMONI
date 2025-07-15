import json
import numpy as np
import os
from config import USERS_FILE, DISTANCE_THRESHOLD
import sqlite3


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Erreur de décodage JSON dans le fichier des utilisateurs.")
    return {}

def save_user(user_data):
    with open(USERS_FILE, "w") as f:
        try:
            json.dump(user_data, f, indent=4)
        except TypeError as e:
            print(f"Erreur lors de l'enregistrement de l'utilisateur: {e}")
            

def retrieve_memory(user_id, conn):
    if conn is None:
        raise ValueError("A database connection is required.")
    
    cursor = conn.cursor()
    cursor.execute(f"SELECT name, description, tags, value FROM {user_id}")
    rows = cursor.fetchall()

    memory = []
    for row in rows:
        name, description, tags, value = row
        memory.append({
            "name": name,
            "description": description,
            "tags": tags.split(";") if tags else [],
            "value": value
        })
    
    return memory

def user_retriever(encodings, conn):
    if conn is None:
        raise ValueError("A database connection is required.")
    
    cursor = conn.cursor()

    # Check if the user already exists based on embeddings
    cursor.execute("SELECT user_id, embeddings FROM user_embeddings WHERE embeddings IS NOT NULL")

    for user_id, blob in cursor.fetchall():
        stored = np.frombuffer(blob, dtype=np.float32)
        distance = np.linalg.norm(np.array(encodings[0]) - stored)

        if distance < DISTANCE_THRESHOLD:
            try:
                user_memory= retrieve_memory(user_id, conn)

            except sqlite3.OperationalError:
                user_memory = ""

            return user_id, user_memory

    # if no user found, create a new user
    cursor.execute("SELECT COUNT(*) FROM user_embeddings")
    user_number = cursor.fetchone()[0] + 1
    new_user_id = f"user{user_number}"
    user_memory = ""

    embeddings_blob = np.array(encodings[0], dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT INTO user_embeddings (user_id, embeddings) VALUES (?, ?)",
        (new_user_id, embeddings_blob)
    )

    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS {new_user_id} (name TEXT, description TEXT, tags TEXT, value TEXT, embeddings BLOB)"
    )

    conn.commit()
    return new_user_id, user_memory
