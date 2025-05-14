from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import List, Dict, Union

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------
# Seed for Reproducibility
# -----------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)

# -----------------------
# Rounding Utility
# -----------------------
def traditional_round(number: float, ndigits: int = 0) -> Union[int, float]:
    multiplier = 10 ** ndigits
    rounded = (number * multiplier + 0.5 if number >= 0 else number * multiplier - 0.5)
    result = rounded // 1 / multiplier
    return int(result) if ndigits == 0 else result

# -----------------------
# Knowledge Vector Classes
# -----------------------
class KVector:
    def __init__(self, num_bins: int = 10) -> None:
        self.num_bins = num_bins
        self.vector = [0.5] * num_bins

    def mean(self) -> float:
        total_mass = sum(self.vector)
        if total_mass == 0:
            return 0.5
        return sum(((i + 0.5) / self.num_bins) * mass for i, mass in enumerate(self.vector)) / total_mass

    def update(self, difficulty: float, correct: bool, update_size: float = 1.0) -> None:
        threshold = self.num_bins * difficulty
        if correct:
            idxs = [i for i in range(self.num_bins) if (i + 0.5) >= threshold]
        else:
            idxs = [i for i in range(self.num_bins) if (i + 0.5) <= threshold]
        for i in idxs:
            self.vector[i] += update_size / len(idxs)

class KVectors:
    def __init__(self, categories: List[str]) -> None:
        self.vectors: Dict[str, KVector] = {cat: KVector() for cat in categories}

    def get_estimate(self, category: str) -> float:
        return traditional_round(self.vectors[category].mean(), ndigits=2)

    def update(self, category: str, difficulty: float, correct: bool) -> None:
        self.vectors[category].update(difficulty, correct)

# -----------------------
# Data Loading Functions
# -----------------------
def load_attempts(filepath: str) -> tuple:
    with open(filepath, 'r') as file:
        attempts = [json.loads(line) for line in file]
    train = [(a['user_id'], a['question_id'], a['user_correct']) for a in attempts if a['split'] == 'train']
    val = [(a['user_id'], a['question_id'], a['user_correct']) for a in attempts if a['split'] == 'val']
    test = [(a['user_id'], a['question_id'], a['user_correct']) for a in attempts if a['split'] == 'test']
    return train, val, test

def load_questions(filepath: str) -> tuple:
    with open(filepath, 'r') as f:
        questions = json.load(f)
    qid_to_category = {q['id']: q['category'] for q in questions}
    qid_to_difficulty = {q['id']: q['difficulty'] for q in questions}
    categories = sorted(set(q['category'] for q in questions))
    return qid_to_category, qid_to_difficulty, categories

# -----------------------
# Model Training
# -----------------------
def train_user_vectors(train_data, qid_to_category, qid_to_difficulty, categories):
    user_data = defaultdict(list)
    for uid, qid, correct in train_data:
        user_data[int(uid)].append((int(qid), correct))

    user_vectors = {}
    for user_id, history in user_data.items():
        kvectors = KVectors(categories)
        for qid, correct in history:
            if qid not in qid_to_difficulty:
                continue
            category = qid_to_category[qid]
            difficulty = qid_to_difficulty[qid]
            kvectors.update(category, difficulty, correct)
        user_vectors[user_id] = kvectors
    return user_vectors

# -----------------------
# Evaluation Function
# -----------------------
def evaluate_model(user_vectors, data, qid_to_category, qid_to_difficulty, name="Set"):
    y_true, y_pred = [], []

    for uid, qid, label in data:
        uid, qid = int(uid), int(qid)
        if uid not in user_vectors or qid not in qid_to_difficulty:
            continue
        category = qid_to_category[qid]
        difficulty = qid_to_difficulty[qid]
        estimate = user_vectors[uid].get_estimate(category)
        pred = int(estimate >= difficulty)

        y_true.append(label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"--- {name} Evaluation ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print()

# -----------------------
# Main Execution
# -----------------------
def main():
    set_seed(42)
    train_data, val_data, test_data = load_attempts('../data/attempts.jsonl')
    qid_to_category, qid_to_difficulty, categories = load_questions('../data/questions.json')

    user_vectors = train_user_vectors(train_data, qid_to_category, qid_to_difficulty, categories)
    evaluate_model(user_vectors, val_data, qid_to_category, qid_to_difficulty, name="Validation")
    evaluate_model(user_vectors, test_data, qid_to_category, qid_to_difficulty, name="Test")

if __name__ == "__main__":
    main()
