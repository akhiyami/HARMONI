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

import json

def save_user(user_data):
    with open(USERS_FILE, "r+") as f:
        try:
            content = f.read()

            if content.strip() == "{}":
                f.seek(0)
                f.write("{\n")
            else:
                lines = content.splitlines()
                if lines:
                    lines.pop()
                content = "\n".join(lines)
                f.seek(0)
                f.write(content)
                f.truncate()
                f.write(",\n")

            encodings = user_data["encodings"]
            user_data_copy = user_data.copy()
            user_data_copy["encodings"] = "__ENCODINGS__"

            user_json = {f"user{user_data['id']}": user_data_copy}
            json_pretty = json.dumps(user_json, indent=4)
            encodings_str = json.dumps(encodings, separators=(",", ":"))

            final_entry = json_pretty.replace('"__ENCODINGS__"', encodings_str)
            final_entry = final_entry.strip()[1:-1].strip()

            f.write(final_entry)
            f.write("\n}")

        except TypeError as e:
            print(f"Erreur lors de l'enregistrement de l'utilisateur: {e}")

def user_retriever(encodings, users_data):
    for user_id, data in users_data.items():
        stored = data.get("encodings", [])
        if stored:
            distance = np.linalg.norm(np.array(encodings[0]) - np.array(stored))
            if distance < 0.5:
                return user_id

    new_user_id = f"user{len(users_data) + 1}"
    users_data[new_user_id] = {
        "id": f"{len(users_data) + 1}",
        "encodings": (encodings[0]).tolist()
    }
    save_user(users_data[new_user_id])
    return new_user_id

def dump_interaction_history(history, path):
     with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        for i, (user, messages) in enumerate(history.items()):
            f.write(f'    "{user}": [\n')
            for j, msg in enumerate(messages):
                msg_line = json.dumps(msg, ensure_ascii=False)
                comma = "," if j < len(messages) - 1 else ""
                f.write(f"        {msg_line}{comma}\n")
            f.write("    ]" + (",\n" if i < len(history) - 1 else "\n"))
        f.write("}\n")
