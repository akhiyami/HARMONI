"""
Memory management module for user interactions in a conversational AI system.
This module provides functions to retrieve and update user memory in a SQLite database.
It includes functions to retrieve a user's memory, update it with new features, and retrieve or create a user based on face encodings.
==========
Functions:
- memory_retriever: Retrieve the memory of a user from the database.
- update_memory: Update the user's memory in the database with new memory objects.
- user_retriever: Retrieve or create a user based on face encodings.
"""

#--------------------------------------- Imports ---------------------------------------#

import os
import json
import sqlite3

import torch
from transformers import SiglipVisionModel, SiglipImageProcessor
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import cv2

import numpy as np

from config.settings import RECOGNITION_THRESHOLD 
from config.utils import suppress_stdout


#--------------------------------------- Functions ---------------------------------------#

def memory_retriever(user_id, conn):
    """  
    Retrieve the memory of a user from the database.
    This function fetches the user's memory as a list of dictionaries, each containing
    the name and value of each memory object.
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
    cursor.execute(f"SELECT type, name, description, value FROM {user_id}")
    rows = cursor.fetchall()

    # Format the retrieved data into a list of dictionaries
    memory = []
    for row in rows:
        type, name, description, value = row
        memory.append({
            "type": type,
            "name": name,
            "description": description,
            "value": value
        })
    
    return memory


def update_memory(new_memory_object, current_user, conn, database=None):
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
    new_conn = False
    if conn is None:
        conn = sqlite3.connect(database)
        new_conn = True

    primary_memory = new_memory_object.primary_features
    contextual_memory = new_memory_object.features

    cursor = conn.cursor()

    # Update primary features
    cursor.execute(f"SELECT name FROM {current_user} WHERE type = 'primary'")
    existing_primary_features = cursor.fetchall()
    existing_primary_features = [feature[0] for feature in existing_primary_features]
    # check if all the primary features are already in the database
    # If not, insert them
    primary_features = {
        "nom": "Les prénoms et noms de l'utilisateur.",
        "age": "L'âge de l'utilisateur.",
        "genre": "Le genre de l'utilisateur (masculin, féminin, non-binaire, etc.).",
        "preference_dialogue": "Les préférences de dialogue de l'utilisateur (formel, informel, humoristique, etc.)."
    }
    for feature, description in primary_features.items():
        if feature not in existing_primary_features:
            cursor.execute(
                f"INSERT INTO {current_user} (type, name, description, value, embeddings) VALUES (?, ?, ?, ?, ?)",
                ("primary", feature, description, '', None)
            )
            conn.commit()

    new_primary_features = [item for item in primary_memory if item.type == "primary"]

    for item in new_primary_features:
        if item.value:
            cursor.execute(
                f"UPDATE {current_user} SET value = ? WHERE name = ?",
                ((";").join(item.value) if isinstance(item.value, (list, tuple)) else str(item.value), item.name)
            )
            conn.commit()


    # Update contextual features
    cursor.execute(f"SELECT name FROM {current_user} WHERE type = 'contextual'")
    existing_features = cursor.fetchall()
    new_features = [item for item in contextual_memory if item.type == "contextual"]

    for item in new_features:
        if item.value:
            # Check if the feature already exists based on the name
            if item.name not in [feature[0] for feature in existing_features]:  # Feature does not exist
                # Insert new feature into the user's memory
                cursor.execute(
                    f"INSERT INTO {current_user} (type, name, description, value, embeddings) VALUES (?, ?, ?, ?, ?)",
                    (
                        item.type, 
                        item.name, 
                        item.description, 
                        (";").join(item.value), 
                        json.dumps(item.embeddings) if item.embeddings else None
                    )
                )
                conn.commit()
            else:  # Feature exists, update it
                cursor.execute(
                    f"UPDATE {current_user} SET value = ?, embeddings = ? WHERE name = ?",
                    (
                        (";").join(item.value) if isinstance(item.value, (list, tuple)) else str(item.value),
                        json.dumps(item.embeddings) if item.embeddings else None,
                        item.name
                    )
                )
                conn.commit()
    if new_conn:
        conn.close()
    return memory_retriever(current_user, conn)


def get_embeddings(img, model_config):
    if model_config["name"] == "ULIP-p16":
        model = model_config["model"]
        processor = model_config["processor"]

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.pooler_output  # shape: [1, hidden_dim]

        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding
    
    elif model_config["name"] == "INSIGHTFACE":
        model = model_config["model"]
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        with suppress_stdout():
            embedding = model.get_feat(img)

        embedding = torch.tensor(embedding)
        # Normalize the embedding
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding
    
    else:
        raise ValueError("Unknown model name provided for embedding extraction.")

 
def user_retriever(img, conn, model_config, database=None):
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
    new_conn = False
    if conn is None:
        conn = sqlite3.connect(database)
        new_conn = True

    embedding = get_embeddings(img, model_config)

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
        if best_score.item() > 0.5: #TODO: remake this a config variable
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

    # Insert the new user into the user_embeddings table
    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT INTO user_embeddings (user_id, embeddings) VALUES (?, ?)",
        (new_user_id, embedding_blob)
    )

    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS {new_user_id} (type TEXT, name TEXT, description TEXT, value TEXT, embeddings BLOB)"
    )

    #create empty slots for the primary features
    primary_features = {
        "nom": "Les prénoms et noms de l'utilisateur.",
        "age": "L'âge de l'utilisateur.",
        "genre": "Le genre de l'utilisateur (masculin, féminin, non-binaire, etc.).",
        "preference_dialogue": "Les préférences de dialogue de l'utilisateur (formel, informel, humoristique, etc.)."
    }
    for feature, description in primary_features.items():
        cursor.execute(
            f"INSERT INTO {new_user_id} (type, name, description, value, embeddings) VALUES (?, ?, ?, ?, ?)",
            ("primary", feature, description, '', None),
        )

    user_memory = memory_retriever(new_user_id, conn)

    conn.commit()
    if new_conn:
        conn.close()
        
    return new_user_id, user_memory
