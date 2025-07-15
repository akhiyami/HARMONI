from collections import deque
from openai import OpenAI
import openai
from config import API_KEY, LEN_HISTORY

import time

from sklearn.metrics.pairwise import cosine_similarity

import json
import re
import numpy as np

from pydantic import BaseModel, Field
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from llm.prompts import context, qa_instructions

client = OpenAI(api_key=API_KEY)
rag_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


TagsListType = List[str]
ValueListType = List[str]


class Feature(BaseModel):
    name: str = Field(
        ..., 
        pattern=r"^\w+$", 
        description="Doit être un mot unique sans espaces ni caractères spéciaux, décrivant la catégorie d'une caractéristique utilisateur (par exemple: nom, âge, hobby...)."
    )
    description: str = Field(
        ..., 
        description="Description de la caractéristique, pour donner plus de contexte et de détails sur ce qu'elle représente."
    )
    tags: TagsListType = Field(
        ..., 
        description="Liste de mots-clés associés à la caractéristique, pour faciliter la recherche et le filtrage. Doit contenir entre 1 et 3 mots-clés.",
        min_items=1,
        max_items=3
    )
    value: ValueListType = Field(
        ..., 
        description="Liste de valeurs associées à la caractéristique, pour représenter les différentes facettes ou aspects de cette caractéristique.",
        min_items=1
    )
    embeddings: Optional[List[float]] = Field(
        None, 
        description="Représentation vectorielle de la caractéristique, utilisée pour la recherche sémantique et la similarité. Elle sera générée automatiquement plus tard."
    )

class AnswerWithMemory(BaseModel):
    answer: str
    updated_memory: list[Feature]
        

def features_retriever(question, conn, user_id):
    """
    Retrieve relevant features from the memory based on the question.
    """
    
    question_embeddings = rag_model.encode(question, convert_to_tensor=True).cpu().numpy()  
    
    cursor = conn.cursor()

    #Update embeddings for features that do not have them
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
        
        cs = cosine_similarity(
            question_embeddings.reshape(1, -1), 
            np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
        ).flatten()
        cosine_similarities[rowid] = cs[0]

        COSINE_THESHOLD = 0.5
        filtered = [(rowid, sim) for rowid, sim in cosine_similarities.items() if sim > COSINE_THESHOLD]
        
        top_filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:5]
        top_indices = [rowid for rowid, _ in top_filtered]

        if top_indices:
            cursor.execute(
                f"SELECT name, description, tags, value FROM {user_id} WHERE rowid IN ({','.join(['?']*len(top_indices))})", 
                top_indices
            )
            rows = cursor.fetchall()
        else:
            rows = []

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
    stm = deque(history, maxlen=LEN_HISTORY)

    # retrieved only pertinent informations from long term memory
    #RAG
    now = time.time()
    retrieved_memory = features_retriever(question, conn=conn, user_id=current_user)
    retrival_time = time.time() - now
    print(f"Retrieved features: {[feature["name"] for feature in retrieved_memory]}")

    ltm = {
        "role": "system", 
         "content": [{ 
            "type": "text",
            "text": json.dumps(retrieved_memory, indent=2)  
        }]
    }

    now = time.time()
    completion = completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[context, ltm, *stm, {"role": "user", "content": question}],
        response_format=AnswerWithMemory
    )
    generation_time = time.time() - now

    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Retrieval time: {retrival_time:.2f} seconds")
    
    return completion.choices[0].message.parsed


def answer_question(question, memory):
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

   