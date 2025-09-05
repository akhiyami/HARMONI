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

# --- Setup paths ---
root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from conversation.llm.openai_inferences import generate_answer

# --- Load environment variables ---
dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# --- Constants ---
DATABASE_PATH = 'evaluation/persona.db'
CONVERSATIONS_FILES = [
    'general/easy',
    'general/medium',
    'general/hard',
    'spedific/easy',
    'spedific/medium',
    'spedific/hard',
]
OUTPUT_DIR = 'results/persona'

PROMPT_EVAL = {
    "role": "system",
    "content": (
        "Tu es un robot nommé 'QT' qui discute comme un humain, de façon détendue, naturelle et chaleureuse.\n"
        "Ton travail est de renseigner et d'aider les résidents d'une maison de retraite.\n"
        "Tu reçois l'historique de la conversation et une intervention d'un utilisateur.\n"
        "Ta mission est de répondre de manière fluide, engageante et adaptée.\n"
        "Instructions : rédige comme dans une discussion détendue. Sois curieux·se et bienveillant·e. Tout en français.\n"
        "Si tu ne sais pas répondre, redirige l'utilisateur vers un humain.\n"
    ),
}

# --- Functions ---
def load_conversations(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def evaluate_example(example, conn):
    question = example['question']
    ground_truth = example['chosen']
    current_session = []
    context = ""
    current_user = f"user{example['user_id']}"
    visual_profile = {"emotion": None, "gender": None, "age": None}

    # Generate answer using memory
    answer, _ = generate_answer(question, current_session, context, conn, current_user, visual_profile, verbose=False)

    # Evaluate using Rouge and BERT
    rouge = Rouge()
    r_scores = rouge.get_scores(answer, ground_truth)[0]
    P, R, F1 = score([answer], [ground_truth], lang="fr", verbose=False)
    scores_memory = {
        "rouge_1_r": r_scores['rouge-1']['r'], "rouge_1_p": r_scores['rouge-1']['p'], "rouge_1_f1": r_scores['rouge-1']['f'],
        "rouge_2_r": r_scores['rouge-2']['r'], "rouge_2_p": r_scores['rouge-2']['p'], "rouge_2_f1": r_scores['rouge-2']['f'],
        "rouge_l_r": r_scores['rouge-l']['r'], "rouge_l_p": r_scores['rouge-l']['p'], "rouge_l_f1": r_scores['rouge-l']['f'],
        "bert_r": R.mean().item(), "bert_p": P.mean().item(), "bert_f1": F1.mean().item()
    }

    # Generate answer without memory
    answer_no_memory = client.chat.completions.create(
        model="gpt-4o",
        messages=[PROMPT_EVAL, *current_session, {"role": "user", "content": question}],
    ).choices[0].message.content

    r_scores_no_memory = rouge.get_scores(answer_no_memory, ground_truth)[0]
    P_nm, R_nm, F1_nm = score([answer_no_memory], [ground_truth], lang="fr", verbose=False)
    scores_no_memory = {
        "rouge_1_r": r_scores_no_memory['rouge-1']['r'], "rouge_1_p": r_scores_no_memory['rouge-1']['p'], "rouge_1_f1": r_scores_no_memory['rouge-1']['f'],
        "rouge_2_r": r_scores_no_memory['rouge-2']['r'], "rouge_2_p": r_scores_no_memory['rouge-2']['p'], "rouge_2_f1": r_scores_no_memory['rouge-2']['f'],
        "rouge_l_r": r_scores_no_memory['rouge-l']['r'], "rouge_l_p": r_scores_no_memory['rouge-l']['p'], "rouge_l_f1": r_scores_no_memory['rouge-l']['f'],
        "bert_r": R_nm.mean().item(), "bert_p": P_nm.mean().item(), "bert_f1": F1_nm.mean().item()
    }

    return question, answer, answer_no_memory, ground_truth, scores_memory, scores_no_memory

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

        scores_memory_list = []
        scores_no_memory_list = []
        qa_records = []

        for example in tqdm(conversations):
            q, a, a_nm, gt, scores_mem, scores_nm = evaluate_example(example, conn)
            scores_memory_list.append(scores_mem)
            scores_no_memory_list.append(scores_nm)
            qa_records.append({"question": q, "answer_memory": a, "answer_no_memory": a_nm, "ground_truth": gt})

        # Save Q/A for this file
        save_qa(file.replace("/", "_"), qa_records)

        # Save evaluation results
        df_memory = pd.DataFrame(scores_memory_list)
        df_no_memory = pd.DataFrame(scores_no_memory_list)
        averages_memory = df_memory.mean().to_dict()
        averages_no_memory = df_no_memory.mean().to_dict()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_results = pd.DataFrame([averages_memory, averages_no_memory], index=['our_method', 'gpt4o'])
        df_results.to_csv(os.path.join(OUTPUT_DIR, f"{file.replace('/', '_')}_evaluation.csv"))

        # Print summary
        print("\nOUR METHOD (with memory):")
        print(df_memory.mean())
        print("\nGPT-4o METHOD (no memory):")
        print(df_no_memory.mean())

if __name__ == "__main__":
    main()
