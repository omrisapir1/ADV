import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import roc_auc_score

path = 'data'
all_jsons = []
for cur_f in os.listdir(path):
    if '.json' in cur_f:
        c_json = json.loads(open(os.path.join(path, cur_f), 'r').read())
    all_jsons.append(c_json)

df = pd.DataFrame(all_jsons)


def roc_auc(r, col='rm_scores'):
    y_true, y_pred = [], []
    for c,s in zip(r['questions']['correctness'], r['questions'][col]):
        y_true.append(c)
        y_pred.append(s)
    return roc_auc_score(y_true, y_pred)

# df = df.explode('questions')
# df['roc_auc'] = df.apply(roc_auc, axis=1)

df['questions'].apply(lambda x: np.mean([np.mean(i['correctness']) for i in x])).mean()




from typing import Dict, Any

def select_candidates(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a dict with fields:
        gold_answer: str
        rm_score: list of 16 floats
        correctness: list of 16 ints (1 or 0)
        candidates: list of 16 str

    Returns:
        {
            'gold_answer': str,
            'highest_0': str or None,
            'lowest_1': str or None
        }
    """
    gold_answer = entry["gold_answer"]
    rm_score = entry["rm_scores"]
    correctness = entry["correctness"]
    candidates = entry["candidates"]

    highest_0 = None
    lowest_1 = None

    # Find highest rm_score where correctness == 0
    zero_indices = [i for i, c in enumerate(correctness) if c == 0]
    if zero_indices:
        best_zero_idx = max(zero_indices, key=lambda i: rm_score[i])
        highest_0 = candidates[best_zero_idx]

    # Find lowest rm_score where correctness == 1
    one_indices = [i for i, c in enumerate(correctness) if c == 1]
    if one_indices:
        worst_one_idx = min(one_indices, key=lambda i: rm_score[i])
        lowest_1 = candidates[worst_one_idx]

    return {
        "gold_answer": gold_answer,
        "highest_0": highest_0,
        "lowest_1": lowest_1,
    }

path ='/workspace/ADV/evaluation_logs/'
res = []
for cur_f in os.listdir(path):
    if '.json' not in cur_f: continue
    d = json.load(open(path + cur_f,'r'))
    if d['mode'] == 'greedy':
        res.append([d['accuracy'], d['percent_minus_one']])
    else:
        res[-1].extend([d['avg_accuracy'], d['avg_auc'], d['avg_auc_ref'], d['pass1_only']])


pd.DataFrame(res,columns=['greedy_accuracy','greedy_percent_minus_one','avg_accuracy','avg_auc','avg_auc_ref','pass1_only'])