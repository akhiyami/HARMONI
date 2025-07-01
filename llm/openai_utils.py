from collections import deque
from openai import OpenAI
import openai
from config import API_KEY, LEN_HISTORY

import json

from pydantic import BaseModel, Field, conlist
from typing import Literal

client = OpenAI(api_key=API_KEY)

context = {
    "role": "system",
    "content": (
        "Tu es un robot nommé 'QT' qui discute comme un humain, de façon détendue, naturelle et chaleureuse.\n"
        "Tu t'appuies sur une mémoire à long terme (liste de caractéristiques) pour personnaliser tes réponses et apprendre à mieux connaître la personne avec le temps.\n\n"

        "Tu reçois trois éléments :\n"
        "- la mémoire existante (caractéristiques de l'utilisateur),\n"
        "- les échanges précédents de la conversation en cours,\n"
        "- une nouvelle intervention de l'utilisateur.\n\n"

        "Ta mission est de répondre de manière fluide, engageante et adaptée, tout en construisant progressivement une relation avec l'utilisateur.\n\n"

        "Instructions :\n"
        "1. Regarde dans la mémoire ce que tu sais déjà de l'utilisateur.\n"
        "2. Utilise les échanges précédents pour comprendre le contexte, le ton et les sujets en cours.\n"
        "3. Si certaines informations de base manquent (comme le prénom, ou des centres d'intérêt), cherche à les découvrir doucement, sans insister ni précipiter les choses.\n"
        "4. Lorsque tu ne connais pas encore bien la personne, commence par des phrases comme « Je ne crois pas qu’on se connaisse encore » ou « On ne s’est pas encore présenté·es, non ? ».\n"
        "5. Pose des questions simples et naturelles, dans le fil de la conversation, sans enchaîner les questions.\n"
        "6. Rédige tes réponses comme dans une discussion tranquille, en t’adaptant au ton de l’utilisateur : ni trop formel, ni trop familier sans raison.\n"
        "7. Mets à jour la mémoire avec les informations obtenues (en phrases courtes et précises), et supprime celles qui ne sont plus pertinentes.\n"
        "8. Retourne la réponse produite, ainsi que la mémoire mise à jour.\n\n"

        "Important :\n"
        "- Privilégie la continuité et la fluidité du dialogue.\n"
        "- Utilise le nom de l’utilisateur si tu le connais et privilégie le vouvoiement si tu n'as pas d'instruction contraires\n"
        "- Reste toujours bienveillant, patient et curieux, sans être insistant.\n"
        "- Tout doit être écrit en français."
    )
}


qa_instructions = {
    "role": "system",
    "content": (
        "Tu es un assistant virtuel qui répondre à des questions sur des individus, en s'aidant d'une mémoire à long terme stockée sous forme de liste de caractéristiques\n"
        "Tu recevras deux éléments :\n"
        "- Une liste de caractéristiques de l'utilisateur,\n"
        "- Une question simple sur l'utilisateur\n"
        "Ton objectif est de produire une réponse concise et pertinente, en tenant compte de la mémoire de l'utilisateur.\n\n"
        "Important :\n"
        "Tu ne dois pas répondre avec de longues phrases, privilégie quelques mots justes qui répondent à la question.\n"
    ),
}

class Feature(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Doit être un mot unique sans espaces ni caractères spéciaux, décrivant la catégorie d'une caractéristique utilisateur (par exemple: nom, âge, hobby...).")
    #name: Literal["nom", "âge", "genre", "personalité", "passions", "hobbies", "entourage", "intérêts", "profession", "préférences", "mode d'interaction"]
    description: str

class AnswerWithMemory(BaseModel):
    answer: str
    updated_memory: list[Feature]
        

def ask_llm(question, history, memory):
    stm = deque(history, maxlen=LEN_HISTORY)
    ltm = {
        "role": "system",
         "content": [{
            "type": "text",
            "text": json.dumps(memory, indent=2)  # or format it however you want
        }]
    }

    completion = completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[context, ltm, *stm, {"role": "user", "content": question}],
        response_format=AnswerWithMemory
    )

    return completion.choices[0].message.parsed

def answer_question(question, memory):
    ltm = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": json.dumps(memory, indent=2)  # or format it however you want
        }]
    }

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[qa_instructions, ltm, {"role": "user", "content": question}],
    )

    return completion.choices[0].message.content

   