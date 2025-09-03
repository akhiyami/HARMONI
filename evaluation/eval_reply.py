import sqlite3

import json 
import sys
import os
import dotenv
from tqdm import tqdm

import pandas as pd
from rouge import Rouge 
from bert_score import score

from openai import OpenAI

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from conversation.llm.openai_inferences import generate_answer

prompt_eval ={
    "role": "system",
    "content": (
        "Tu es un robot nommé 'QT' qui discute comme un humain, de façon détendue, naturelle et chaleureuse.\n"
        "Ton travail est de renseigner et d'aider les résidents d'une maison de retraite.\n"

        "Tu reçois :\n"
        "- l'historique de la conversation,\n"        
        "- Une intervention d'un utilisateur.\n\n"

        "Ta mission est de répondre de manière fluide, engageante et adaptée.\n\n"

        "Instructions :\n"
        "- Rédige comme dans une discussion détendue. Sois curieux·se, bienveillant·e, et adapte ton ton aux utilisateurs.\n"
        "- Ne pose pas trop de questions à la suite, mais intègre-les naturellement.\n"
        "- Tout doit être écrit en français. \n\n"
        "- Tu peux ne pas avoir la réponse à une question. Dans ce cas, tu dois rediriger l'utilisateur vers un humain, et lui dire que tu ne sais pas.\n"

        "Important :\n"
        "- Tu peux converser avec plusieurs utilisateurs en même temps, mais tu dois toujours te souvenir de qui est qui.\n"
        "- Privilégie des réponses courtes et simples, comme dans une conversation normale.\n"
    ),
}

dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

database = 'tests/persona.db'
conn = sqlite3.connect(database)

conversations_files= [
    'general/easy',
    'general/medium',
    'general/hard',
    'spedific/easy',
    'spedific/medium',
    'spedific/hard',
]

for file in conversations_files:
    print(f"Processing {file}.jsonl")

    conversation_path = f'/Users/isir_acide/Documents/Context-aware_Conversation/data/PersonaFeedback/data/{file}.jsonl'

    with open(conversation_path, 'r') as f:
        conversations = [json.loads(line) for line in f]

    rouge = Rouge()

    def eval_example(exemple, conn):
        question = exemple['question']
        ground_truth = exemple['chosen']

        current_session = []
        context = ""

        current_user = f"user{exemple['user_id']}"

        visual_profile = {
            "emotion": None,
            "gender": None,
            "age": None,
        }
                        
        answer, _ = generate_answer(question, current_session, context, conn, current_user, visual_profile, verbose=False)

        r_scores = rouge.get_scores(answer, ground_truth)[0]
        P, R, F1 = score([answer], [ground_truth], lang="fr", verbose=False)

        scores_memory_method =  {
            "rouge_1_r": r_scores['rouge-1']['r'],
            "rouge_1_p": r_scores['rouge-1']['p'],
            "rouge_1_f1": r_scores['rouge-1']['f'],
            "rouge_2_r": r_scores['rouge-2']['r'],
            "rouge_2_p": r_scores['rouge-2']['p'],
            "rouge_2_f1": r_scores['rouge-2']['f'],
            "rouge_l_r": r_scores['rouge-l']['r'],
            "rouge_l_p": r_scores['rouge-l']['p'],
            "rouge_l_f1": r_scores['rouge-l']['f'],
            "bert_r": R.mean().item(),
            "bert_p": P.mean().item(),
            "bert_f1": F1.mean().item()
        }

        # now try the no memory method (generate directly with gpt 4o API client)

        answer_no_memory = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                prompt_eval,
                *current_session,
                {"role": "user", "content": question}
            ],
        ).choices[0].message.content

        print("Question:", question)
        print("Our Answer:", answer)
        print("Answer (No Memory):", answer_no_memory)
        print("Ground Truth:", ground_truth)

        r_scores_no_memory = rouge.get_scores(answer_no_memory, ground_truth)[0]
        P_no_memory, R_no_memory, F1_no_memory = score([answer_no_memory], [ground_truth], lang="fr", verbose=False)
        scores_no_memory_method =  {
            "rouge_1_r": r_scores_no_memory['rouge-1']['r'],
            "rouge_1_p": r_scores_no_memory['rouge-1']['p'],
            "rouge_1_f1": r_scores_no_memory['rouge-1']['f'],
            "rouge_2_r": r_scores_no_memory['rouge-2']['r'],
            "rouge_2_p": r_scores_no_memory['rouge-2']['p'],
            "rouge_2_f1": r_scores_no_memory['rouge-2']['f'],
            "rouge_l_r": r_scores_no_memory['rouge-l']['r'],
            "rouge_l_p": r_scores_no_memory['rouge-l']['p'],
            "rouge_l_f1": r_scores_no_memory['rouge-l']['f'],
            "bert_r": R_no_memory.mean().item(),
            "bert_p": P_no_memory.mean().item(),
            "bert_f1": F1_no_memory.mean().item()
        }
        
        return scores_memory_method, scores_no_memory_method

    scores_memory = []
    scores_no_memory = []
    for example in tqdm(conversations):
        result_memory, result_no_memory = eval_example(example, conn)
        scores_memory.append(result_memory)
        scores_no_memory.append(result_no_memory)

    df_memory = pd.DataFrame(scores_memory)
    df_no_memory = pd.DataFrame(scores_no_memory)

    # Print results
    print("OUR RESULTS WITH USER PROFILE \n")

    print("ROUGE scores:")
    print("==========")
    print(f"Average Rouge 1: {df_memory['rouge_1_f1'].mean():.4f}")
    print(f"Average Rouge 2: {df_memory['rouge_2_f1'].mean():.4f}")
    print(f"Average Rouge l: {df_memory['rouge_l_f1'].mean():.4f}")
    print("\nBERT scores:")
    print("==========")
    print(f"Average BERT R: {df_memory['bert_r'].mean():.4f}")
    print(f"Average BERT P: {df_memory['bert_p'].mean():.4f}")
    print(f"Average BERT F1: {df_memory['bert_f1'].mean():.4f}")

    averages_memory = {
        'rouge_1_r': df_memory['rouge_1_r'].mean(),
        'rouge_1_p': df_memory['rouge_1_p'].mean(),
        'rouge_1_f1': df_memory['rouge_1_f1'].mean(),
        'rouge_2_r': df_memory['rouge_2_r'].mean(),
        'rouge_2_p': df_memory['rouge_2_p'].mean(),
        'rouge_2_f1': df_memory['rouge_2_f1'].mean(),
        'rouge_l_r': df_memory['rouge_l_r'].mean(),
        'rouge_l_p': df_memory['rouge_l_p'].mean(),
        'rouge_l_f1': df_memory['rouge_l_f1'].mean(),
        'bert_r': df_memory['bert_r'].mean(),
        'bert_p': df_memory['bert_p'].mean(),
        'bert_f1': df_memory['bert_f1'].mean()
    }

    print("\n\nGPT 4o RESULTS\n")

    print("ROUGE scores:")
    print("==========")
    print(f"Average Rouge 1: {df_no_memory['rouge_1_f1'].mean():.4f}")
    print(f"Average Rouge 2: {df_no_memory['rouge_2_f1'].mean():.4f}")
    print(f"Average Rouge l: {df_no_memory['rouge_l_f1'].mean():.4f}")
    print("\nBERT scores:")
    print("==========")
    print(f"Average BERT R: {df_no_memory['bert_r'].mean():.4f}")
    print(f"Average BERT P: {df_no_memory['bert_p'].mean():.4f}")
    print(f"Average BERT F1: {df_no_memory['bert_f1'].mean():.4f}")

    averages_no_memory = {
        'rouge_1_r': df_no_memory['rouge_1_r'].mean(),
        'rouge_1_p': df_no_memory['rouge_1_p'].mean(),
        'rouge_1_f1': df_no_memory['rouge_1_f1'].mean(),
        'rouge_2_r': df_no_memory['rouge_2_r'].mean(),
        'rouge_2_p': df_no_memory['rouge_2_p'].mean(),
        'rouge_2_f1': df_no_memory['rouge_2_f1'].mean(),
        'rouge_l_r': df_no_memory['rouge_l_r'].mean(),
        'rouge_l_p': df_no_memory['rouge_l_p'].mean(),
        'rouge_l_f1': df_no_memory['rouge_l_f1'].mean(),
        'bert_r': df_no_memory['bert_r'].mean(),
        'bert_p': df_no_memory['bert_p'].mean(),
        'bert_f1': df_no_memory['bert_f1'].mean()
    }

    # Save results to log file
    csv_path = f'results/persona/{file}_evaluation.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_results = pd.DataFrame([averages_memory, averages_no_memory], index=['our_method', 'gpt4o'])
    df_results.to_csv(csv_path)