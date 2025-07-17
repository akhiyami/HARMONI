"""
Memory management module for user interactions in a conversational AI system.
"""

##########
# Import necessary libraries
##########

import os
import json
import sqlite3

import torch
from transformers import SiglipVisionModel, SiglipImageProcessor
from PIL import Image

import numpy as np

from config import RECOGNITION_THRESHOLD

##########
# Functions
##########

def memory_retriever(user_id, conn):
    """  
    Retrieve the memory of a user from the database.
    This function fetches the user's memory as a list of dictionaries, each containing
    the name, description, tags, and value of each memory object.
    -----
    Args:
        user_id (str): The unique identifier for the user.
        conn (sqlite3.Connection): The database connection object.
    Returns:
        list: A list of dictionaries representing the user's memory.
    """
    if conn is None:
        raise ValueError("A database connection is required.")
    
    # Retrieve the user's features from its table in the database
    cursor = conn.cursor()
    cursor.execute(f"SELECT name, description, tags, value FROM {user_id}")
    rows = cursor.fetchall()

    # Format the retrieved data into a list of dictionaries
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


def update_memory(new_memory_object, current_user, conn):
    """
    Update the user's memory in the database with new memory objects.
    This function checks if the feature already exists and updates it if it does,
    or inserts it if it does not.
    -----
    Args:
        new_memory_object (list): A list of memory objects to be added or updated.
        current_user (str): The unique identifier for the user.
        conn (sqlite3.Connection): The database connection object.
    Returns:
        list: The updated memory of the user.   
    """
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM {current_user}")
    existing_features = cursor.fetchall()

    for item in new_memory_object:
        # Check if the feature already exists based on the name
        if item.name not in [feature[0] for feature in existing_features]:  # Feature does not exist
            # Insert new feature into the user's memory
            cursor.execute(
                f"INSERT INTO {current_user} (name, description, tags, value) VALUES (?, ?, ?, ?)",
                (item.name, item.description, (";").join(item.tags), (";").join(item.value))
            )
            conn.commit()
        else:  # Feature exists, update it
            cursor.execute(
                f"UPDATE {current_user} SET description = ?, tags = ?, value = ? WHERE name = ?",
                (item.description, (";").join(item.tags), (";").join(item.value), item.name)
            )
            conn.commit()

    return memory_retriever(current_user, conn)


def user_retriever(img, conn, processor, model):
    """
    Retrieve or create a user based on face encodings.
    This function checks if the user already exists in the database based on the provided encodings.
    If a user is found, it returns the user ID and their memory.
    If no user is found, it creates a new user with the provided encodings and the associated table, and returns the new user ID and an empty memory.
    -----
    Args:
        encodings (list): A list of face encodings for the user.
        conn (sqlite3.Connection): The database connection object.
    Returns:
        tuple: A tuple containing the user ID and their memory.
    """
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output  # shape: [1, hidden_dim]

    embedding = torch.nn.functional.normalize(embedding, dim=-1)

    if conn is None:
        raise ValueError("A database connection is required.")
    cursor = conn.cursor()

    # Check if the user already exists based on embeddings
    cursor.execute("SELECT user_id, embeddings FROM user_embeddings WHERE embeddings IS NOT NULL")
    embeddings_dict = {user_id: np.frombuffer(blob, dtype=np.float32) for user_id, blob in cursor.fetchall()}

    known_user_embeddings = list(embeddings_dict.values())
    if known_user_embeddings != []:
        known_user_embeddings = torch.stack([torch.tensor(e) for e in known_user_embeddings])

        cos = torch.nn.CosineSimilarity(dim=-1)
        similarities = cos(embedding, known_user_embeddings)

        best_score, best_idx = torch.max(similarities, dim=0)
        if best_score.item() > RECOGNITION_THRESHOLD:
            # If a user is found, return the user ID and their memory
            user_id = list(embeddings_dict.keys())[best_idx.item()]
            try:
                user_memory = memory_retriever(user_id, conn)
            except sqlite3.OperationalError:
                user_memory = ""
            return user_id, user_memory

    # if no user found, create a new user
    cursor.execute("SELECT COUNT(*) FROM user_embeddings")
    user_number = cursor.fetchone()[0] + 1
    new_user_id = f"user{user_number}"
    user_memory = ""

    # Insert the new user into the user_embeddings table
    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT INTO user_embeddings (user_id, embeddings) VALUES (?, ?)",
        (new_user_id, embedding_blob)
    )

    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS {new_user_id} (name TEXT, description TEXT, tags TEXT, value TEXT, embeddings BLOB)"
    )

    conn.commit()
    return new_user_id, user_memory
