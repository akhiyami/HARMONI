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
import sqlite3

import numpy as np
from openai import OpenAI

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from config.settings import API_KEY, LEN_HISTORY, NUM_TOP_FEATURES
from memory.models import LongTermMemory, FeaturesNames
from llm.prompts import qa_instructions, add_feature_prompt, modify_feature_prompt, reply_prompt, feature_identification_prompt
from llm.retriever import features_retriever

# Initialize OpenAI client and RAG model
client = OpenAI(api_key=API_KEY)

#--------------------------------------- Functions for interacting with the LLM ---------------------------------------#

##############################
# ANSWER GENERATION FUNCTION #
##############################

def generate_answer(question, history, context, conn=None, current_user=None, visual_profile=None, verbose=True):
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
    now = time.time()
    retrieved_memory = features_retriever(question, conn=conn, user_id=current_user)
    retrival_time = time.time() - now
    retrieved_features_names = [feature["name"] for feature in retrieved_memory]
    if verbose:
        print(f"Retrieved features: {retrieved_features_names}")

    # Check and manage if a visual profile is provided
    if visual_profile:
        if "emotion" in visual_profile and visual_profile["emotion"] is not None:
            emotion = visual_profile["emotion"]
            retrieved_memory.insert(4, {
                "type": "contextual",
                "name": "emotion",
                "description": "The detected emotion of the user.",
                "tags": ["emotion"],
                "value": [emotion]
            })

        if retrieved_memory[1]["value"] is not None: #age
            if 'age' in visual_profile and visual_profile["age"] is not None:
                age = visual_profile["age"]
                retrieved_memory[1]["value"] = [age]
        
        #check if the age is known in the retrieved memory
        if retrieved_memory[2]["value"] is not None: #gender
            if 'gender' in visual_profile and visual_profile["gender"] is not None:
                gender = visual_profile["gender"]
                retrieved_memory[2]["value"] = [gender]

    # Prepare the long term memory context
    ltm = {
        "role": "system", 
        "content": [{ 
            "type": "text",
            "text": json.dumps(retrieved_memory, indent=2, ensure_ascii=False)  # ensure_ascii=False preserves characters like "é"
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
    completion = client.chat.completions.create(
        model="gpt-4o",
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

def update_memory_llm(user_question, conn=None, current_user=None, database=None):
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

    # stm = deque(history, maxlen=LEN_HISTORY)
    
    ##########
    # Prepare the long term memory context
    ##########
    
    ltm = {
        "primary_features": [],
        "features": []
    }

    # Add the primary features
    cursor = conn.cursor()
    cursor.execute(f"SELECT name, value FROM {current_user} WHERE type = 'primary'")
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
        ltm["primary_features"].append(feature)

    # Add the contextual features
    cursor.execute(f"SELECT name, description, tags, value FROM {current_user} WHERE type = 'contextual'")
    rows = cursor.fetchall()
    for row in rows:    
        name, description, tags, value = row
        feature = {
            "type": "contextual",
            "name": name,
            "description": description,
            "tags": tags.split(";") if tags else [],
            "value": value.split(";") if value else []
        }
        ltm["features"].append(feature)

    # Format the long term memory for the LLM
    ltm ={
        "role": "system", 
         "content": [{ 
            "type": "text",
            "text": json.dumps(ltm, indent=2, ensure_ascii=False)  # ensure_ascii=False preserves characters like "é"
        }]
    }

    ##########
    # Generate the memory update using the LLM
    ##########

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            feature_identification_prompt,
            ltm,
            # *stm,
            {"role": "user", "content": f"Utilisateur : {user_question}"},
        ],
        response_format=FeaturesNames
    )

    features_names = completion.choices[0].message.parsed
    new_features = features_names.Add 
    new_features = [new_feature.name for new_feature in new_features]
    modified_features = features_names.Modify

    #look for the features to modify in db
    to_modify_features = []
    for feature in modified_features:
        cursor.execute(f"SELECT type, name, description, tags, value FROM {current_user} WHERE name = ?", (feature,))
        to_modify_features = cursor.fetchall()

    updated_ltm = LongTermMemory(primary_features=[], features=[])

    import concurrent.futures

    def add_feature_task(new_feature):
        add_feature = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                add_feature_prompt,
                {"role": "user", "content": f"Le nom de la feature à ajouter est : {new_feature}."},
                # *stm,
                {"role": "user", "content": f"Utilisateur : {user_question}"},
            ],
            response_format=LongTermMemory
        )
        return add_feature.choices[0].message.parsed.features[0]

    def modify_feature_task(feature_to_modify):
        modify_feature = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                modify_feature_prompt,
                {"role": "user", "content": f"La feature à modifier est : {json.dumps(feature_to_modify, indent=2, ensure_ascii=False)}."},
                # *stm,
                {"role": "user", "content": f"Utilisateur : {user_question}"},
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
            updated_ltm.features.append(new_feature_parsed)

        # Collect modify_feature results
        for future in concurrent.futures.as_completed(modify_futures):
            modified_feature_parsed = future.result()
            if modified_feature_parsed.primary_features:
                updated_ltm.primary_features.append(modified_feature_parsed.primary_features[0])
            elif modified_feature_parsed.features:
                updated_ltm.features.append(modified_feature_parsed.features[0])

    if new_conn:
        conn.close()

    return updated_ltm

###########################

def answer_question(question, memory):
    """
    Answer a question using the LLM with the provided memory.
    This function prepares the context and sends the request to the LLM.
    -----
    Args:
        question (str): The question to ask the LLM.
        memory (list): The user's memory to be used as context.
    Returns:
        str: The answer from the LLM.
    """
    ltm = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": json.dumps(memory, indent=2, ensure_ascii=False)  
        }]
    }

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[qa_instructions, ltm, {"role": "user", "content": question}],
    )

    return completion.choices[0].message.content

   