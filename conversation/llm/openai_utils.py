"""
This module contains utility functions for interacting with OpenAI's API.
It includes functions for asking questions to the LLM and retrieving answers,
as well as managing user memory.
It also handles the retrieval of relevant features from the user's memory.
"""

#--------------------------------------- Imports ---------------------------------------#

import json
import time
from collections import deque
import sys
import os
import sqlite3

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from config.settings import API_KEY, LEN_HISTORY, NUM_TOP_FEATURES
from memory.models import LongTermMemory
from llm.prompts import qa_instructions, memory_update_prompt, reply_prompt

##########

# Initialize OpenAI client and RAG model
client = OpenAI(api_key=API_KEY)
rag_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

#--------------------------------------- Functions for interacting with the LLM ---------------------------------------#

###############################
# FEATURES RETRIEVER FUNCTION #
###############################

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
            tags_list = tags.split(";")
            embedding = rag_model.encode(", ".join(tags_list)).tolist()
            embedding = np.array(embedding, dtype=np.float32).tobytes()  #Convert to bytes for BLOB storage
            cursor.execute(
                f"UPDATE {user_id} SET embeddings = ? WHERE rowid = ?",
                (embedding, rowid)
            )
            conn.commit()
        
        # Calculate cosine similarity between the question and the feature embeddings
        cs = cosine_similarity(
            question_embeddings.reshape(1, -1), 
            np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
        ).flatten()
        cosine_similarities[rowid] = cs[0]
        
        # Filter features based on a cosine similarity threshold
        COSINE_THESHOLD = 0.5
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


##############################
# ANSWER GENERATION FUNCTION #
##############################

def generate_answer(question, history, context, conn=None, current_user=None, visual_profile=None):
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
            "text": json.dumps(retrieved_memory, indent=2)  
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

    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Retrieval time: {retrival_time:.2f} seconds")
    
    # Parse the response from the LLM
    return completion.choices[0].message.content, retrieved_features_names


##########################
# MEMORY UPDATE FUNCTION #
##########################

def update_memory_llm(user_question, history, conn=None, current_user=None, database=None):
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

    stm = deque(history, maxlen=LEN_HISTORY)
    
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
            "text": json.dumps(ltm, indent=2)  
        }]
    }

    ##########
    # Generate the memory update using the LLM
    ##########

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            memory_update_prompt,
            ltm,
            *stm,
            {"role": "user", "content": f"Utilisateur : {user_question}"},
        ],
        response_format=LongTermMemory
    )

    if new_conn:
        conn.close()
        
    return completion.choices[0].message.parsed


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
            "text": json.dumps(memory, indent=2)  
        }]
    }

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[qa_instructions, ltm, {"role": "user", "content": question}],
    )

    return completion.choices[0].message.content

   