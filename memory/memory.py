import json
import numpy as np
import os
from config import USERS_FILE

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

def user_retriever(encodings, users_data):
    for user_id, data in users_data.items():
        stored = data.get("encodings", [])
        if stored:
            distance = np.linalg.norm(np.array(encodings[0]) - np.array(stored))
            if distance < 0.4:
                return user_id, data.get("user_memory", "")

    new_user_id = f"user{len(users_data) + 1}"
    users_data[new_user_id] = {
        "id": f"{len(users_data) + 1}",
        "encodings": (encodings[0]).tolist()
    }
    save_user(users_data)
    user_history = ""

    return new_user_id, user_history
