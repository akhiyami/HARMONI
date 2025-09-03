import os

import sys
import re
import json
import sqlite3

import dotenv

import pandas as pd
from tqdm import tqdm

from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

from openai import OpenAI

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from conversation.llm.openai_inferences import update_memory_llm
from conversation.memory.memory import update_memory 

conn = sqlite3.connect("locomo.db")

dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)


def generate_gpt(question):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )
    return response.choices[0].message.content

def initialize_speaker_tables(speaker_a, speaker_b, speaker_a_id, speaker_b_id, conn):
    cursor = conn.cursor()

    #reset all tables
    cursor.execute(f"DROP TABLE IF EXISTS {speaker_a_id}")
    cursor.execute(f"DROP TABLE IF EXISTS {speaker_b_id}")

    for speaker_id, name in zip([speaker_a_id, speaker_b_id], [speaker_a, speaker_b]):
        cursor.execute(
        f"CREATE TABLE IF NOT EXISTS {speaker_id} (type TEXT, name TEXT, description TEXT, tags TEXT, value TEXT, embeddings BLOB)"
    )

        #create empty slots for the primary features
        for feature in ["nom", "age", "genre", "preference_dialogue"]:
            value = name if feature == "nom" else None
            cursor.execute(
            f"INSERT INTO {speaker_id} (type, name, description, tags, value) VALUES (?, ?, ?, ?, ?)",
            ("primary", feature, None, None, value),
        )

    conn.commit()

def extract_rouge_statistics(rouge_scores):
    rows = []
    for entry in rouge_scores:
        d = entry[0]  # because each element is a list of dicts
        row = {}
        for metric, scores in d.items():
            for stat, value in scores.items():
                row[f"{metric}_{stat}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    data = df.mean()
    return data

def process_speaker_interactions(
    speaker_name, 
    speaker_id, 
    speaker_interactions, 
    other_interactions, 
    observations, 
    previous_observations, 
    rouge_score, 
    model, 
    conn
):
    """Process one speaker's interactions for a given session and return results."""

    rouge_scores = []
    turn_similarity = []
    observations_session_llm = []
    n_observations_llm = 0
    missed_observations = 0

    cursor = conn.cursor()

    stm = []
    if int(speaker_interactions[0][1][-1]) > 1:
        stm.append({"role": "assistant", "content": f"{other_interactions[0][0]}"})
        other_interactions.pop(0)

    for question, dia_id in tqdm(speaker_interactions, desc="Processing interactions", leave=False):
        # Update memory
        new_memory_blob = update_memory_llm(question, conn=conn, current_user=speaker_id)
        memory_user = update_memory(new_memory_blob, speaker_id, conn)

        features = new_memory_blob.primary_features + new_memory_blob.features

        # Gold observations for this turn
        observation_turn = [obs[0] for obs in observations if obs[1] == dia_id]
        observations_session_llm += [f"Nom: {speaker_name}"]

        observations_session_llm += [
            f"{feature.name}: {feature.value}"
            for feature in features if feature.value
        ]

        if features:
            n_observations_llm += 1
            observation_llm = ". ".join([
                feature.value if isinstance(feature.value, str) else ". ".join(feature.value)
                for feature in features if feature.value
            ])

            if observation_turn:
                if len(observation_turn) > 1:
                    observation_turn = ". ".join(observation_turn)
                else:
                    observation_turn = observation_turn[0]

                # Compute ROUGE
                rouge_scores.append(rouge_score.get_scores(observation_llm, observation_turn))

                # Compute similarity
                emb1 = model.encode(observation_llm, convert_to_tensor=True)
                emb2 = model.encode(observation_turn, convert_to_tensor=True)
                turn_similarity.append(util.cos_sim(emb1, emb2).item())

        elif observation_turn:
            missed_observations += 1

        stm.append({"role": "user", "content": f"{question}"})

        if len(other_interactions) > 0:
            stm.append({"role": "assistant", "content": f"{other_interactions[0][0]}"})
            other_interactions.pop(0)


        #clear profile:
        cursor.execute(f"DELETE FROM {speaker_id}")

        #create empty slots for the primary features
        for feature in ["nom", "age", "genre", "preference_dialogue"]:
            value = speaker_name if feature == "nom" else None
            cursor.execute(
            f"INSERT INTO {speaker_id} (type, name, description, tags, value) VALUES (?, ?, ?, ?, ?)",
            ("primary", feature, None, None, value),
        )
        
        conn.commit()

        

    return {
        "rouge_scores": rouge_scores,
        "turn_similarity": turn_similarity,
        "observations_session_llm": observations_session_llm,
        "n_observations_llm": n_observations_llm,
        "missed_observations": missed_observations,
        "memory_user": memory_user
    }


def summarize_session(session, observations, observations_session_llm, model):
    """Generate GPT summaries and compute session-level similarity."""
    session_observations_gold = ".\n".join([obs[0] for obs in observations])
    session_observations_llm = ".\n".join(observations_session_llm)

    prompt = "Ecris une courte présentation de l'individu décrit par ces observations: {}"
    description_gold = generate_gpt(prompt.format(session_observations_gold))
    description_llm = generate_gpt(prompt.format(session_observations_llm))

    emb1 = model.encode(description_gold, convert_to_tensor=True)
    emb2 = model.encode(description_llm, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2).item()

    return {
        "gold": description_gold,
        "llm": description_llm,
        "similarity": similarity
    }


def process_all_sessions(example, speaker_name, speaker_id, sessions, rouge_score, model, conn):
    """Run processing for all sessions for one speaker."""
    scores_df = pd.DataFrame()
    previous_observations = []
    sessions_observations = {}

    rouge_scores_total = []

    for session in tqdm(sessions, desc=f"Processing {speaker_name}"):
        conversation = example["conversation"][session]
        speaker_interactions = [
            [turn["text"], turn["dia_id"]]
            for turn in conversation if turn["speaker"] == speaker_name
        ]

        other_interactions = [
            [turn["text"], turn["dia_id"]]
            for turn in conversation if turn["speaker"] != speaker_name
        ]

        # Gold observations for session
        observations = example["observation"][f"{session}_observation"][speaker_name]
        previous_observations += observations

        # Process speaker interactions
        res = process_speaker_interactions(
            speaker_name, speaker_id, speaker_interactions, other_interactions,
            observations, previous_observations, rouge_score, model, conn
        )

        mean_rouge = extract_rouge_statistics(res["rouge_scores"])
        turn_similarity_mean = (
            sum(res["turn_similarity"]) / len(res["turn_similarity"])
            if res["turn_similarity"] else 0
        )

        # Summarize session
        summary = summarize_session(session, observations, res["observations_session_llm"], model)
        sessions_observations[session] = {"gold": summary["gold"], "llm": summary["llm"]}

        # Aggregate session-level stats
        data = mean_rouge
        data["observations_similarity"] = turn_similarity_mean
        data["n_turns"] = len(speaker_interactions)
        data["ratio"] = res["n_observations_llm"] / len(observations)
        data["missed_observations"] = res["missed_observations"] / len(observations)
        data["session_similarity"] = summary["similarity"]

        scores_df = pd.concat([scores_df, data.to_frame().T], ignore_index=True)

    return scores_df, sessions_observations


if __name__ == "__main__":

    path = "evaluations/data/locomo_fr.json"

    with open(path, "r") as f:
        locomo_data = json.load(f)

    os.makedirs("evaluations/results", exist_ok=True)

    for id in range(0, 1):
        rouge_score = Rouge()
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        example = locomo_data[id]
        sessions = [k for k in example["conversation"].keys() if re.match(r"session_\d+$", k)]

        speaker_a = example["conversation"]["speaker_a"]
        speaker_b = example["conversation"]["speaker_b"]

        speaker_a_id = f"user{id+0}"
        speaker_b_id = f"user{id+1}"

        conn = sqlite3.connect("locomo.db")
        initialize_speaker_tables(speaker_a, speaker_b, speaker_a_id, speaker_b_id, conn)

        # Process both speakers
        scores_df_a, sessions_observations_a = process_all_sessions(
            example, speaker_a, speaker_a_id, sessions, rouge_score, model, conn
        )
        scores_df_a.to_csv(f"evaluations/results/scores_{id}_a.csv", index=False)
        with open(f"evaluations/results/sessions_observations_{id}_a.json", "w") as f:
            json.dump(sessions_observations_a, f, ensure_ascii=False, indent=4)

        
        scores_df_b, sessions_observations_b = process_all_sessions(
            example, speaker_b, speaker_b_id, sessions, rouge_score, model, conn
        )
        scores_df_b.to_csv(f"evaluations/results/scores_{id}_b.csv", index=False)
        with open(f"evaluations/results/sessions_observations_{id}_b.json", "w") as f:
            json.dump(sessions_observations_b, f, ensure_ascii=False, indent=4)