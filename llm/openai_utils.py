from collections import deque
from openai import OpenAI
from config import API_KEY, LEN_HISTORY

client = OpenAI(api_key=API_KEY)

context = {
    "role": "system",
    "content": (
        "Tu es un assistant virtuel qui peut discuter et répondre \n"
        "Tu sais de ton interlocuteur que: \n"
        "- Il préfère les réponses concises.\n"
    ),
}

def ask_llm(question, history, user_id):
    if user_id not in history:
        history[user_id] = []
    stm = deque(history[user_id], maxlen=LEN_HISTORY)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[context, *stm, {"role": "user", "content": question}],
    )

    return completion.choices[0].message.content.strip()

