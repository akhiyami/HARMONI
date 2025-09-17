import os
import sys
import json
import sqlite3
import dotenv
from tqdm import tqdm
import pandas as pd
from rouge import Rouge
from bert_score import score
from openai import OpenAI
from pydantic import BaseModel
import time
import yaml

# --- Setup paths ---
root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_folder_path)

from conversation.llm.openai_inferences import generate_answer

# --- Load environment variables ---
dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

config_path = os.path.join(root_folder_path, 'config', 'config.yaml')
config = yaml.safe_load(open(config_path))

model = config['reply-llm']['model']
if model.startswith("gpt") and "oss" not in model:
    llm_client = OpenAI(api_key=API_KEY)
else:
    llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

judge_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
judge_model = "mixtral"

# --- Constants ---
DATABASE_PATH = 'evaluation/persona.db'
CONVERSATIONS_FILES = [
    # 'general/easy',
    # 'general/medium',
    # 'general/hard',
    # 'spedific/easy',
    'spedific/medium',
    'spedific/hard',
]
OUTPUT_DIR = 'results/persona'

PROMPT_EVAL = {
    "role": "system",
    "content": (
        "Tu es un robot nommé 'QT' qui discute comme un humain, de façon détendue, naturelle et chaleureuse.\n"
        "Tu reçois l'historique de la conversation et une intervention d'un utilisateur.\n"
        "Ta mission est de répondre de manière fluide, engageante et adaptée.\n"

        "Instructions :\n"
        "- Rédige comme dans une discussion détendue. Sois curieux·se, bienveillant·e, et adapte ton ton aux utilisateurs.\n"
        "- Ne pose pas trop de questions à la suite, mais intègre-les naturellement.\n"
        "- Tout doit être écrit en français. \n\n"

        "Important :\n"
        "- Privilégie des réponses courtes et simples, comme dans une conversation normale.\n"
    ),
}

REPLY_PROMPT_EVAL = {
    "role": "system",
    "content": (
        "Tu es un robot nommé 'QT' qui discute comme un humain, de façon détendue, naturelle et chaleureuse.\n"
        "Tu t'appuies sur une mémoire structurée pour personnaliser tes réponses.\n"
        "- `primary_features` : nom, âge, genre, ton préféré, etc.\n"
        "- `features` : goûts, souvenirs, opinions, centres d'intérêt, etc.\n\n"

        "Tu reçois :\n"
        "- un contexte général de ta situation actuelle, avec les informations que tu es censé connaitre,\n"
        "- des `primary_features`,\n"
        "- des `features` jugées pertinentes,\n"
        "- l'historique de la conversation,\n"
        "- une nouvelle intervention d'un utilisateur.\n\n"

        "Ta mission est de répondre de manière fluide, engageante et adaptée, sans modifier la mémoire.\n\n"

        "Instructions :\n"
        "- Appuie-toi d'abord sur les `primary_features` pour personnaliser ta réponse.\n"
        "- Utilise les `features` si elles sont pertinentes pour enrichir la réponse.\n"
        "- Rédige comme dans une discussion détendue. Sois curieux·se, bienveillant·e, et adapte ton ton aux utilisateurs.\n"
        "- Ne pose pas trop de questions à la suite, mais intègre-les naturellement.\n"
        "- Ne modifie pas la mémoire ici, ta tâche est seulement de répondre à l'utilisateur.\n"
        "- Tout doit être écrit en français. \n\n"
        "- N'invente pas d'informations, base-toi uniquement sur le contexte, et sur le profil des usagers. \n"

        "Important :\n"
        "- Privilégie des réponses courtes et simples, comme dans une conversation normale.\n"
    ),
}

# --- System prompt for LLM-as-judge ---
PROMPT_JUDGE_SYSTEM = {
    "role": "system",
    "content": (
        "Tu es un évaluateur expert de réponses textuelles. "
        "Tu dois évaluer trois réponses à la même question, et fournir pour chacune un score de 0 à 10 "
        "(0 = complètement inappropriée, 10 = excellente)"
        "Tu peux utiliser des exemples de bonnes réponses (7-10) et de mauvaises réponses (0-3) comme référence."
        "Retourne ta réponse au format json suivant : "
        '{"score_a": int, '
        '"score_b": int, '
        '"score_c": int}'
    ),
}


PROMPT_CONTENT = """
Exemple:
Profile: {user_profile}
Question: {question}
Bonne réponse: {chosen}
Mauvaise réponse: {rejected}


Maintenant, évalue ces deux réponses pour la question ci-dessous en donnant un score de0-10 pour chacune:
Profile: {user_profile}
Question: {question}
Réponse A (avec mémoire améliorée): {answer_memory_improved}
Réponse B (avec mémoire): {answer_memory}
Réponse C (sans mémoire): {answer_no_memory}
"""

class ReponseFormat(BaseModel):
    score_a: int
    score_b: int
    score_c: int

# --- Functions ---
def load_conversations(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def evaluate_with_llm(question, answer_memory_improved, answer_memory, answer_no_memory, chosen, rejected, user_profile):
    """
    Uses GPT API to score both answers (0-10) and generate reasoning, given chosen/rejected examples.
    """
    prompt_content = PROMPT_CONTENT.format(
        question=question,
        chosen=chosen,
        rejected=rejected,
        user_profile=json.dumps(user_profile, ensure_ascii=False),
        answer_memory_improved=answer_memory_improved,
        answer_memory=answer_memory,
        answer_no_memory=answer_no_memory
    )

    response = judge_client.chat.completions.parse(
        model=judge_model,
        messages=[PROMPT_JUDGE_SYSTEM, {"role": "user", "content": prompt_content}],
        response_format=ReponseFormat
    )

    return response.choices[0].message.parsed

def evaluate_example(example, conn):
    question = example['question']
    chosen = example['chosen']
    rejected = example['reject']
    current_session = []
    context = ""
    current_user = f"user{example['user_id']}"
    visual_profile = {"emotion": None, "gender": None, "age": None}

    # Generate answer using memory
    start_time = time.time()
    answer_retriever, _ = generate_answer(question, current_session, context, conn, current_user, visual_profile, reply_prompt=REPLY_PROMPT_EVAL, verbose=False)
    time_retriever = time.time() - start_time

    start_time = time.time()
    answer, _ = generate_answer(question, current_session, context, conn, current_user, visual_profile, reply_prompt=REPLY_PROMPT_EVAL, retriever=False, verbose=False)
    time_memory = time.time() - start_time

    # Generate answer without memory
    start_time = time.time()
    answer_no_memory = llm_client.chat.completions.create(
        model=model,
        messages=[PROMPT_EVAL, *current_session, {"role": "user", "content": question}],
    ).choices[0].message.content
    time_no_memory = time.time() - start_time

    cursor = conn.cursor()
    user_profile = {}
    cursor.execute(f"SELECT name, value FROM {current_user}")
    for row in cursor.fetchall():
        user_profile[row[0]] = row[1]

    # LLM scoring
    llm_judge_output = evaluate_with_llm(question, answer_retriever, answer, answer_no_memory, chosen, rejected, user_profile=user_profile)

    # Evaluate using Rouge and BERT
    rouge = Rouge()
    r_scores_mem_ret = rouge.get_scores(answer_retriever, chosen)[0]
    P, R, F1 = score([answer_retriever], [chosen], lang="fr", verbose=False)
    scores_memory_ret = {
        "rouge_1_r": r_scores_mem_ret['rouge-1']['r'], "rouge_1_p": r_scores_mem_ret['rouge-1']['p'], "rouge_1_f1": r_scores_mem_ret['rouge-1']['f'],
        "rouge_2_r": r_scores_mem_ret['rouge-2']['r'], "rouge_2_p": r_scores_mem_ret['rouge-2']['p'], "rouge_2_f1": r_scores_mem_ret['rouge-2']['f'],
        "rouge_l_r": r_scores_mem_ret['rouge-l']['r'], "rouge_l_p": r_scores_mem_ret['rouge-l']['p'], "rouge_l_f1": r_scores_mem_ret['rouge-l']['f'],
        "bert_r": R.mean().item(), "bert_p": P.mean().item(), "bert_f1": F1.mean().item()
    }

    r_scores_mem = rouge.get_scores(answer, chosen)[0]
    P, R, F1 = score([answer], [chosen], lang="fr", verbose=False)
    scores_memory = {
        "rouge_1_r": r_scores_mem['rouge-1']['r'], "rouge_1_p": r_scores_mem['rouge-1']['p'], "rouge_1_f1": r_scores_mem['rouge-1']['f'],
        "rouge_2_r": r_scores_mem['rouge-2']['r'], "rouge_2_p": r_scores_mem['rouge-2']['p'], "rouge_2_f1": r_scores_mem['rouge-2']['f'],
        "rouge_l_r": r_scores_mem['rouge-l']['r'], "rouge_l_p": r_scores_mem['rouge-l']['p'], "rouge_l_f1": r_scores_mem['rouge-l']['f'],
        "bert_r": R.mean().item(), "bert_p": P.mean().item(), "bert_f1": F1.mean().item()
    }

    r_scores_nm = rouge.get_scores(answer_no_memory, chosen)[0]
    P_nm, R_nm, F1_nm = score([answer_no_memory], [chosen], lang="fr", verbose=False)
    scores_no_memory = {
        "rouge_1_r": r_scores_nm['rouge-1']['r'], "rouge_1_p": r_scores_nm['rouge-1']['p'], "rouge_1_f1": r_scores_nm['rouge-1']['f'],
        "rouge_2_r": r_scores_nm['rouge-2']['r'], "rouge_2_p": r_scores_nm['rouge-2']['p'], "rouge_2_f1": r_scores_nm['rouge-2']['f'],
        "rouge_l_r": r_scores_nm['rouge-l']['r'], "rouge_l_p": r_scores_nm['rouge-l']['p'], "rouge_l_f1": r_scores_nm['rouge-l']['f'],
        "bert_r": R_nm.mean().item(), "bert_p": P_nm.mean().item(), "bert_f1": F1_nm.mean().item()
    }

    return {
        "question": question,
        "answer_retriever": answer_retriever,
        "answer_memory": answer,
        "answer_no_memory": answer_no_memory,
        "ground_truth_chosen": chosen,
        "ground_truth_rejected": rejected,
        "llm_score_memory_retriever": llm_judge_output.score_a,
        "llm_score_memory": llm_judge_output.score_b,
        "llm_score_no_memory": llm_judge_output.score_c,
        "scores_memory_retriever": scores_memory_ret,
        "scores_memory": scores_memory,
        "scores_no_memory": scores_no_memory,
        "time_memory_retriever": time_retriever,
        "time_memory": time_memory,
        "time_no_memory": time_no_memory
    }

def save_qa(file_name, qa_data):
    qa_dir = os.path.join(OUTPUT_DIR, "qa")
    os.makedirs(qa_dir, exist_ok=True)
    output_path = os.path.join(qa_dir, f"{file_name}_qa.json")
    with open(output_path, 'w') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

def main():
    conn = sqlite3.connect(DATABASE_PATH)

    for file in CONVERSATIONS_FILES:
        print(f"\nProcessing {file}.jsonl")
        conversation_path = f'/Users/isir_acide/Documents/Context-aware_Conversation/data/PersonaFeedback/data/{file}.jsonl'
        conversations = load_conversations(conversation_path)

        scores_memory_ret_list = [] 
        scores_memory_list = []
        scores_no_memory_list = []
        qa_records = []
        llm_scores_memory_ret = []
        llm_scores_memory = []
        llm_scores_no_memory = []
        times_memory_ret = []
        times_memory = []
        times_no_memory = []

        for example in tqdm(conversations):
            result = evaluate_example(example, conn)
            scores_memory_ret_list.append(result['scores_memory_retriever'])
            scores_memory_list.append(result['scores_memory'])
            scores_no_memory_list.append(result['scores_no_memory'])
            llm_scores_memory_ret.append(result['llm_score_memory_retriever'])
            llm_scores_memory.append(result['llm_score_memory'])
            llm_scores_no_memory.append(result['llm_score_no_memory'])
            times_memory_ret.append(result['time_memory_retriever'])
            times_memory.append(result['time_memory'])
            times_no_memory.append(result['time_no_memory'])
            qa_records.append(result)

        # Save Q/A for this file
        save_qa(file.replace("/", "_"), qa_records)

        # Save evaluation results
        df_memory_ret = pd.DataFrame(scores_memory_ret_list)
        df_memory = pd.DataFrame(scores_memory_list)
        df_no_memory = pd.DataFrame(scores_no_memory_list)
        df_memory_ret['llm_score'] = llm_scores_memory_ret
        df_memory['llm_score'] = llm_scores_memory
        df_no_memory['llm_score'] = llm_scores_no_memory
        df_memory_ret['time'] = times_memory_ret
        df_memory['time'] = times_memory
        df_no_memory['time'] = times_no_memory
        averages_memory_ret = df_memory_ret.mean().to_dict()
        averages_memory = df_memory.mean().to_dict()
        averages_no_memory = df_no_memory.mean().to_dict()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_results = pd.DataFrame([averages_memory_ret, averages_memory, averages_no_memory], index=['memory with retriever', 'memory', 'base'])
        df_results.to_csv(os.path.join(OUTPUT_DIR, f"{file.replace('/', '_')}_evaluation.csv"))


if __name__ == "__main__":
    main()
