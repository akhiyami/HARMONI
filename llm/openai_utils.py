from collections import deque
from openai import OpenAI
import openai
from config import API_KEY, LEN_HISTORY

from pydantic import BaseModel

client = OpenAI(api_key=API_KEY)

context = {
    "role": "system",
    "content": (
        "Tu es un assistant virtuel qui peut discuter et répondre, en s'aidant d'une mémoire à long terme stockée sous forme de liste de caractéristiques\n"
        "Tu recevras trois éléments :\n"
        "- une mémoire existante (liste de traits de personnalité),\n"
        "- les précédentes interactions de la discussion en cours.\n"
        "- une question de l'utilisateur.\n\n"
        "Ton objectif est de produire une réponse concise et pertinente, en tenant compte de la mémoire de l'utilisateur.\n\n"

        "Instructions à suivre :\n"
        "1. Analyse la mémoire existante pour comprendre les traits de personnalité de l'utilisateur.\n"
        "2. Étudie les interactions précédentes pour saisir le contexte de la discussion.\n"
        "3. Réponds à la question de l'utilisateur en intégrant les informations de la mémoire et du contexte.\n"
        "4. Rédige une réponse claire et concise, en utilisant un langage simple et direct.\n"
        "5. Mets à jour la mémoire de l'utilisateur en fonction des nouvelles informations pertinentes fournies dans la réponse.\n"
        "6. Retourne la réponse, ainsi que la mémoire mise à jour sous forme de liste de traits de personnalité.\n\n"
        "Important :\n"
        "La mémoire doit aider à maintenir la continuité du dialogue et permettre des réponses adaptées à la personnalité de l'utilisateur.\n"
    ),
}

qa_instructions = {
    "role": "system",
    "content": (
        "Tu es un assistant virtuel qui peut discuter et répondre, en s'aidant d'informtions stockée sous forme de mémoire à long terme \n"
        "Tu réponds ici en quelques mots, sans faire de phrase, dans le cadre d'un quizz.\n"
    ),
}

memory_instructions = {
    "role": "system",
    "content": (
        "Tu es un modèle de langage avancé, capable de stocker et de mettre à jour une mémoire contenant les traits de personnalité de l'utilisateur.\n"
        "Tu recevras deux éléments :\n"
        "- une mémoire existante (liste de traits de personnalité),\n"
        "- un nouveau contexte de dialogue.\n\n"
        "Ton objectif est de produire une mémoire mise à jour, intégrant les nouvelles informations pertinentes.\n\n"
        "Instructions à suivre :\n"
        "1. Analyse la mémoire existante et identifie les traits de personnalité déjà connus.\n"
        "2. Étudie le nouveau contexte de dialogue pour repérer toute nouvelle information de personnalité ou tout changement.\n"
        "3. Fusionne les anciennes et nouvelles informations pour produire une représentation actualisée de la personnalité de l'utilisateur.\n"
        "4. Rédige la mémoire mise à jour sous forme de liste à puces claire et concise (maximum 20 points), en utilisant des phrases simples.\n\n"
        "Important :\n"
        "La mémoire doit aider à maintenir la continuité du dialogue et permettre des réponses adaptées à la personnalité de l'utilisateur."
    ),
}

class Feature(BaseModel):
    name: str
    description: str

class AnswerWithMemory(BaseModel):
    answer: str
    updated_memory: list[Feature]
        

def ask_llm(question, history, memory):
    stm = deque(history, maxlen=LEN_HISTORY)
    ltm = {
        "role": "system",
        "content": memory,
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
        "content": memory,
    }

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[context, ltm, {"role": "user", "content": question}],
    )

    return completion.choices[0].message.content


def update_memory(old_memory, new_memory):
    
    formatted_memory = [f"{item["role"]}: {item["content"]}" for item in new_memory]
    old_context = {
        "role": "user",
        "content": f"Mémoire existante : \n{old_memory} \nContexte de dialogue : \n" + "\n".join(formatted_memory),
    }

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[memory_instructions, old_context],
    )

    user_context = completion.choices[0].message.content.strip()
    return user_context
    
   