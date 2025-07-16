"""
This module contains utility functions for interacting with OpenAI's API.
It includes functions for asking questions to the LLM and retrieving answers,
as well as managing user memory.
It also handles the retrieval of relevant features from the user's memory.
"""

############
# Import necessary libraries
############

import json
import re
import time
from collections import deque

import numpy as np
from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

from config import API_KEY, LEN_HISTORY, NUM_TOP_FEATURES
from llm.prompts import context

##########

# Initialize OpenAI client and RAG model
client = OpenAI(api_key=API_KEY)
rag_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


###########
# Type Definitions
###########

TagsListType = List[str]
ValueListType = List[str]

# Define a Pydantic model for the feature structure
class Feature(BaseModel):
    name: str = Field(
        ..., 
        pattern=r"^\w+$", 
        description="Must be a single word without spaces or special characters, describing the category of a user feature (e.g., name, age, hobby...)."
    )
    description: str = Field(
        ..., 
        description="Description of the feature, providing more context and details about what it represents."
    )
    tags: TagsListType = Field(
        ..., 
        description="List of keywords associated with the feature, to facilitate search and filtering. Must contain between 1 and 3 keywords.",
        min_items=1,
        max_items=3
    )
    value: ValueListType = Field(
        ..., 
        description="List of values associated with the feature, representing the different facets or aspects of this feature.",
        min_items=1
    )
    embeddings: Optional[List[float]] = Field(
        None, 
        description="Vector representation of the feature, used for semantic search and similarity. It will be generated automatically later."
    )

# Define a Pydantic model for the answer with updated memory
class AnswerWithMemory(BaseModel):
    answer: str
    updated_memory: list[Feature]
        
############
# Function Definitions
############

def features_retriever(question, conn, user_id):
    """
    Retrieve relevant features from the memory based on the question.
    """
    # Convert the question into embeddings
    question_embeddings = rag_model.encode(question, convert_to_tensor=True).cpu().numpy()  

    #Update embeddings for features that do not have them
    cursor = conn.cursor()
    cursor.execute(f"SELECT rowid, tags, embeddings FROM {user_id}")
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
    memory = []
    for row in rows:
        name, description, tags, value = row
        feature = {
            "name": name,
            "description": description,
            "tags": tags.split(";") if tags else [],
            "value": value.split(";")
        }
        memory.append(feature)

    return memory


def ask_llm(question, history, conn=None, current_user=None):
    """
    Ask the LLM a question and retrieve the answer along with updated memory.
    This function prepares the context and session history, retrieves relevant features from the user's memory,
    and sends the request to the LLM.
    -----
    Args:
        question (str): The question to ask the LLM.
        history (deque): The session history containing previous interactions.
        conn (sqlite3.Connection, optional): The database connection object. Defaults to None.
        current_user (str, optional): The unique identifier for the user. Defaults to None.
    Returns:
        AnswerWithMemory: The answer from the LLM along with updated memory.
    """
    stm = deque(history, maxlen=LEN_HISTORY)

    # retrieved only pertinent informations from long term memory
    now = time.time()
    retrieved_memory = features_retriever(question, conn=conn, user_id=current_user)
    retrival_time = time.time() - now
    print(f"Retrieved features: {[feature["name"] for feature in retrieved_memory]}")

    # Prepare the long term memory context
    ltm = {
        "role": "system", 
         "content": [{ 
            "type": "text",
            "text": json.dumps(retrieved_memory, indent=2)  
        }]
    }

    now = time.time()
    # Generate the answer using the LLM
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[context, ltm, *stm, {"role": "user", "content": question}],
        response_format=AnswerWithMemory
    )
    generation_time = time.time() - now

    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Retrieval time: {retrival_time:.2f} seconds")
    
    # Parse the answer and updated memory from the completion
    return completion.choices[0].message.parsed

