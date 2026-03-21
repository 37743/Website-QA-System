import json
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config

config = get_config()

MODEL_NAME = config['model']['embed_model']

print(f"Loading tokenizer and model for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=int(config['embedding']['max_length'])
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = mean_pooling(outputs, inputs['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze().tolist()

def build_match_text(match):
    # Mapping the specific keys from your JSON structure
    team_a = match.get("team_a", "فريق أ")
    team_b = match.get("team_b", "فريق ب")
    score_a = match.get("score_a", "0")
    score_b = match.get("score_b", "0")
    date = match.get("page_date", "") # Changed from 'date'
    time = match.get("page_time", "") # Changed from 'time'
    competition = match.get("competition", "")
    round_name = match.get("round", "")

    parts = [
        f"مباراة في {competition} ({round_name}) بين {team_a} و {team_b}.",
        f"النتيجة النهائية هي {score_a} ل{team_a} مقابل {score_b} ل{team_b}."
    ]

    if date:
        parts.append(f"أقيمت المباراة بتاريخ {date}.")
    if time:
        parts.append(f"في تمام الساعة {time}.")

    # Processing Scorers
    scorers = match.get("scorers", {})
    for team, p_list in scorers.items():
        if p_list:
            team_name = team_a if team == "team_a" else team_b
            goals = [f"{p.get('player_name')} في الدقيقة {p.get('goal_time')}" for p in p_list]
            parts.append(f"أهداف {team_name}: " + "، ".join(goals) + ".")

    # Processing Stats
    stats = match.get("stats", {})
    if stats:
        stat_summary = []
        for key, val in stats.items():
            stat_summary.append(f"{key} ({val.get('team_a')} - {val.get('team_b')})")
        parts.append("إحصائيات (فريق أ - فريق ب): " + " | ".join(stat_summary) + ".")

    # Processing Events (Handling the 'description' field)
    events = match.get("events", [])
    if events:
        event_descriptions = []
        for e in events:
            desc = e.get("description", "").strip()
            minute = e.get("minute", "")
            if desc: # Skip empty descriptions (like referees at 0')
                event_descriptions.append(f"دقيقة {minute}: {desc}")
        
        if event_descriptions:
            parts.append("أهم أحداث المباراة: " + " | ".join(event_descriptions) + ".")

    return " ".join(parts)

def process_embeddings(input_json, output_json):
    print("Generating embeddings...")
    
    # Load data - handling potential list or single object
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        if isinstance(data, dict): # If the file is a single object, wrap in list
            data = [data]

    with open(output_json, 'w', encoding='utf-8') as outfile:
        for match in data:
            text = build_match_text(match)
            embedding = get_embedding(text)

            out = {
                "url": match.get("url"),
                "team_a": match.get("team_a"),
                "team_b": match.get("team_b"),
                "score_total": f"{match.get('score_a')}-{match.get('score_b')}",
                "date": match.get("page_date"),
                "text": text,
                "embedding": embedding,
            }

            json.dump(out, outfile, ensure_ascii=False)
            outfile.write("\n")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    input_file = os.path.join(base_dir, 'data_collection', 'datacollection1', 'output', 'scrap.json')
    output_file = os.path.join(base_dir, 'embedding', 'output', 'embedded_data.json')

    process_embeddings(input_file, output_file)
    print("Embeddings saved.")