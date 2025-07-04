from collections import deque
from openai import OpenAI
import openai
from config import API_KEY, LEN_HISTORY

import time

from sklearn.metrics.pairwise import cosine_similarity

import json
import re

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
        

def features_retriever(memory, question):
    """
    Retrieve relevant features from the memory based on the question.
    """
    if memory is None or len(memory) == 0:
        return []

    question_embeddings = rag_model.encode(question, convert_to_tensor=True)

    for feature in memory:
        if "embeddings" not in feature.keys() or feature["embeddings"] is None:
            feature["embeddings"] = rag_model.encode((", ".join(feature["tags"]))).tolist()  

    memory_embeddings = [feature["embeddings"] for feature in memory]

    cosine_similarities = cosine_similarity(
        question_embeddings.reshape(1, -1).cpu().numpy(), 
        memory_embeddings
    ).flatten()  

    n_indices = min(5, len(memory))  
    top_indices = cosine_similarities.argsort()[-n_indices:][::-1] 

    retrieved_memory = [memory[i] for i in top_indices if cosine_similarities[i] > 0.5]  

    #remove embeddings from retrieved memory
    for feature in retrieved_memory:
        feature.pop("embeddings", None)

    return retrieved_memory

def ask_llm(question, history, memory):
    stm = deque(history, maxlen=LEN_HISTORY)

    # retrieved only pertinent informations from long term memory
    #RAG

    now = time.time()
    retrieved_memory = features_retriever(memory, question)
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

   