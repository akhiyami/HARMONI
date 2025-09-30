"""
This module contains utility functions for interacting with OpenAI's API.
It includes functions for asking questions to the LLM and retrieving answers,
as well as managing user memory.
It also handles the retrieval of relevant features from the user's memory.
"""

#--------------------------------------- Imports ---------------------------------------#

import json
from pyexpat import features
import time
from collections import deque
import sys
import os
import yaml
import sqlite3
import concurrent.futures

import numpy as np
from openai import OpenAI

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from config.settings import API_KEY, LEN_HISTORY
from memory.models import LongTermMemory, FeaturesNames
from llm.prompts import add_feature_prompt, modify_feature_prompt, reply_prompt, feature_identification_prompt
from llm.retriever import features_retriever, attach_embeddings


# Initialize LLM clients 
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

reply_model = config.get("reply-llm", {}).get("model", "")
memory_model = config.get("memory-llm", {}).get("model", "")

use_openai_client = {"reply": False, "memory": False}
if API_KEY:
    openai_client = OpenAI(api_key=API_KEY)
    available_models = [model.id for model in openai_client.models.list().data]
    if reply_model in available_models:
        use_openai_client["reply"] = True
    if memory_model in available_models:
        use_openai_client["memory"] = True

if use_openai_client["reply"]:
    reply_client = OpenAI(api_key=API_KEY)
else: 
    try:
        reply_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    except Exception as e:
        print(f"Error initializing local LLM client.\n{e}")

if use_openai_client["memory"]:
    memory_client = OpenAI(api_key=API_KEY)
else: 
    try:
        memory_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    except Exception as e:
        print(f"Error initializing local LLM client.\n{e}")


#--------------------------------------- Functions for interacting with the LLM ---------------------------------------#

##############################
# ANSWER GENERATION FUNCTION #
##############################

def generate_answer(question, history, context, conn=None, current_user=None, visual_profile=None, reply_prompt=reply_prompt, retriever=True, verbose=True):
    """
    Ask the LLM a question and retrieve the answer along with updated memory.
    This function prepares the context and session history, retrieves relevant features from the user's memory,
    and sends the request to the LLM.
    -----
    Args:
        question (str): The question to ask the LLM.
        history (deque): The session history containing previous interactions.
        context (str): Additional context to provide to the LLM.
        conn (sqlite3.Connection, optional): The database connection object. Defaults to None.
        current_user (str, optional): The unique identifier for the user. Defaults to None.
        visual_profile (dict, optional): A dictionary containing visual profile information
        verbose (bool, optional): Whether to print debug information. Defaults to True.
    Returns:
        str: The answer from the LLM.
    """
    # provide last interactions as short term memory
    stm = deque(history, maxlen=LEN_HISTORY)

    # retrieved only pertinent informations from long term memory
    if retriever:
        now = time.time()
        retrieved_memory = features_retriever(question, conn=conn, user_id=current_user)
        retrival_time = time.time() - now
        retrieved_features_names = [feature["name"] for feature in retrieved_memory]
        if verbose:
            print(f"Retrieved features: {retrieved_features_names}")
        memory = retrieved_memory
    else:
        memory = []
        cursor = conn.cursor()
        retrieved_features_names = []

        cursor.execute(f"SELECT type, name, description, value FROM {current_user}")
        rows = cursor.fetchall()
        for type, name, description, value in rows:
            feature = {
                "type": type,
                "name": name,
                "description": description,
                "value": value.split(";") if value else []
            }
            memory.append(feature)
            retrieved_features_names.append(name)

    # Check and manage if a visual profile is provided
    if visual_profile:
        if "emotion" in visual_profile and visual_profile["emotion"] is not None:
            emotion = visual_profile["emotion"]
            memory.insert(4, {
                "type": "contextual",
                "name": "emotion",
                "description": "Émotion actuelle de l'utilisateur, détectée à partir de son expression faciale.",
                "value": [emotion]
            })

        if memory[1]["value"] is not None: #age
            if 'age' in visual_profile and visual_profile["age"] is not None:
                age = visual_profile["age"]
                memory[1]["value"] = [age]
        
        #check if the age is known in the retrieved memory
        if memory[2]["value"] is not None: #gender
            if 'gender' in visual_profile and visual_profile["gender"] is not None:
                gender = visual_profile["gender"]
                memory[2]["value"] = [gender]

    # Prepare the long term memory context
    ltm = {
        "role": "system", 
        "content": [{ 
            "type": "text",
            "text": json.dumps(memory, indent=2, ensure_ascii=False)  # ensure_ascii=False preserves characters like "é"
        }]
    }

    # Prepare the context prompt
    if context is None:
        context = ''
        
    context_prompt = {
        "role": "system",
        "content": (
            context
        )
    }
    
    # Generate the answer using the LLM
    now = time.time()
    completion = reply_client.chat.completions.create(
        model=reply_model,
        messages=[reply_prompt, context_prompt, ltm, *stm, {"role": "user", "content": question}],
    )
    generation_time = time.time() - now

    if verbose:
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Retrieval time: {retrival_time:.2f} seconds")
    
    # Parse the response from the LLM
    return completion.choices[0].message.content, retrieved_features_names


##########################
# MEMORY UPDATE FUNCTION #
##########################

def update_memory_llm(user_question, conn=None, current_user=None, database=None, stm=None):
    """
    Appelle le LLM pour analyser l’échange et mettre à jour la mémoire.
    Args:
        user_question (str): Dernière question ou phrase de l’utilisateur. 
        history (deque): Historique de session s’il y en a (facultatif).
        conn (sqlite3.Connection, optional): Connexion à la base de données SQLite (facultatif).
        current_user (str, optional): Identifiant unique de l'utilisateur (facultatif).
        database (str): Chemin vers la base de données (facultatif).
    Returns:
        Dict: mémoire mise à jour (avec `primary_features` et `features`)
    """
    # Check if the database connection is provided, otherwise create a new one
    new_conn = False
    if conn is None:
        conn = sqlite3.connect(database)
        new_conn = True

    ##########
    # Last interactions serve as short term memory
    ##########
    if stm: 
        stm = deque(stm, maxlen=LEN_HISTORY)
    else: 
        stm = deque([], maxlen=LEN_HISTORY)
    
    ##########
    # Prepare the long term memory context
    ##########
    
    ltm = {
        "primary_features": [],
        "features": []
    }

    # Add the primary features
    cursor = conn.cursor()
    cursor.execute(f"SELECT name, description, value FROM {current_user} WHERE type = 'primary'")
    rows = cursor.fetchall()
    for row in rows:
        name, description,value = row
        feature = {
            "type": "primary",
            "name": name,
            "description": description,
            "value": value.split(";") if value else []
        }
        ltm["primary_features"].append(feature)

    # Add the contextual features
    cursor.execute(f"SELECT name, description, value FROM {current_user} WHERE type = 'contextual'")
    rows = cursor.fetchall()
    for row in rows:    
        name, description, value = row
        feature = {
            "type": "contextual",
            "name": name,
            "description": description,
            "value": value.split(";") if value else []
        }
        ltm["features"].append(feature)

    # Format the long term memory for the LLM
    ltm = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": json.dumps(ltm, indent=2, ensure_ascii=False)  # ensure_ascii=False preserves characters like "é"
        }]
    }

    ##########
    # Generate the memory update using the LLM
    ##########

    completion = memory_client.chat.completions.parse(
        model=memory_model,
        messages=[
            feature_identification_prompt,
            ltm,
            {"role": "user", "content": f"{user_question}"},
        ],
        response_format=FeaturesNames
    )

    if not completion or not completion.choices or not hasattr(completion.choices[0].message, "parsed"):
        if completion:
            print(f"NOT ABLE TO PARSE THE COMPLETION: {completion}")
        new_features = []
        modified_features = []
    else:
        features_names = completion.choices[0].message.parsed
        new_features = getattr(features_names, "Add", [])
        new_features = [new_feature.name for new_feature in new_features]
        modified_features = getattr(features_names, "Modify", [])
    
    #look for the features to modify in db
    to_modify_features = []
    for feature in modified_features:
        cursor.execute(f"SELECT type, name, description, value FROM {current_user} WHERE name = ?", (feature,))
        to_modify_features = cursor.fetchall()

    updated_ltm = LongTermMemory(primary_features=[], features=[])

    primary_features = {
        "nom": "Les prénoms et noms de l'utilisateur.",
        "age": "L'âge de l'utilisateur.",
        "genre": "Le genre de l'utilisateur (masculin, féminin, non-binaire, etc.).",
        "preference_dialogue": "Les préférences de dialogue de l'utilisateur (formel, informel, humoristique, etc.)."
    }

    def add_feature_task(new_feature):
        if config.get("memory", {}).get("closed_vocabulary", True):
            if new_feature in primary_features:
                description = primary_features[new_feature]
            else:
                description = config.get("memory", {}).get("vocabulary", {}).get(new_feature, "")

        else:
            description = ""
        add_feature = memory_client.chat.completions.parse(
            model=memory_model,
            messages=[
                add_feature_prompt,
                {"role": "system", "content": f"La feature à ajouter est : {new_feature}({description})."},
                {"role": "system", "content": f"Dernières interactions : {stm}."},
                {"role": "user", "content": f"{user_question}"},
            ],
            response_format=LongTermMemory
        )
        return add_feature.choices[0].message.parsed

    def modify_feature_task(feature_to_modify):
        modify_feature = memory_client.chat.completions.parse(
            model=memory_model,
            messages=[
                modify_feature_prompt,
                {"role": "system", "content": f"La feature à modifier est : {json.dumps(feature_to_modify, indent=2, ensure_ascii=False)}."},
                {"role": "system", "content": f"Dernières interactions : {stm}."},
                {"role": "user", "content": f"{user_question}"},
            ],
            response_format=LongTermMemory
        )
        parsed = modify_feature.choices[0].message.parsed
        return parsed

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Launch add_feature tasks in parallel
        add_futures = [executor.submit(add_feature_task, new_feature) for new_feature in new_features]
        # Launch modify_feature tasks in parallel
        modify_futures = [executor.submit(modify_feature_task, feature_to_modify) for feature_to_modify in to_modify_features]

        # Collect add_feature results
        for future in concurrent.futures.as_completed(add_futures):
            new_feature_parsed = future.result()
            if new_feature_parsed.primary_features:
                updated_ltm.primary_features.append(new_feature_parsed.primary_features[0])
            elif new_feature_parsed.features:
                if new_feature_parsed.features[0].value:
                    updated_ltm.features.append(attach_embeddings(new_feature_parsed.features[0]))

        # Collect modify_feature results
        for future in concurrent.futures.as_completed(modify_futures):
            modified_feature_parsed = future.result()
            if modified_feature_parsed.primary_features:
                updated_ltm.primary_features.append(modified_feature_parsed.primary_features[0])
            elif modified_feature_parsed.features:
                if modified_feature_parsed.features[0].value:
                    updated_ltm.features.append(attach_embeddings(modified_feature_parsed.features[0]))

    if new_conn:
        conn.close()

    return updated_ltm

   