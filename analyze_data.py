import os
import pandas as pd
import json
from sklearn.metrics import roc_auc_score

path = 'data'
all_jsons = []
for cur_f in os.listdir(path):
    if '.json' in cur_f:
        c_json = json.loads(open(os.path.join(path, cur_f), 'r').read())
    all_jsons.append(c_json)

df = pd.DataFrame(all_jsons)
df = df.explode('questions')

def roc_auc(r):
    return roc_auc_score(r['questions']['correctness'], r['questions']['rm_scores'])

df['roc_auc'] = df.apply(roc_auc, axis=1)