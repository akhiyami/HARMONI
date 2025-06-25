from collections import deque
from openai import OpenAI
from config import API_KEY, LEN_HISTORY

client = OpenAI(api_key=API_KEY)

context = {
    "role": "system",
    "content": (
        "Tu es un assistant virtuel qui peut discuter et répondre, en s'aidant d'informtions stockée sous forme de mémoire à long terme \n"
        "Tu privilégies les réponses courtes et concises\n"
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
        

def ask_llm(question, history, memory):
    stm = deque(history, maxlen=LEN_HISTORY)
    ltm = {
        "role": "system",
        "content": memory,
    }

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[context, ltm, *stm, {"role": "user", "content": question}],
    )

    return completion.choices[0].message.content.strip()

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
    
   