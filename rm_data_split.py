from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np

dataset_name = "omrisap/prm_dataset_6k"
seed = 42

# ------------------------
# Load dataset
# ------------------------
ds = load_dataset(dataset_name, split="train")
df = ds.to_pandas()

# ensure consistent random
np.random.seed(seed)

# ------------------------
# 1) Sample 500 for evaluation
# ------------------------
eval_df = df.sample(500, random_state=seed)
remain_df = df.drop(eval_df.index)

# ------------------------
# Helper masks
# ------------------------
q_mask = remain_df["q_wrong_0"].notnull() & (remain_df["q_wrong_0"] != "")
s_mask = remain_df["s_wrong_0"].notnull() & (remain_df["s_wrong_0"] != "")

q_df = remain_df[q_mask]
s_df = remain_df[s_mask]

# ------------------------
# 2) Dataset A
#   - 500 rows w/ q_wrong_0
#   - ALL s_wrong_0 rows EXCEPT 500 reserved
# ------------------------
qA = q_df.sample(500, random_state=seed) if len(q_df) >= 500 else q_df.copy()

# reserve 500 from s_wrong_0 for B
sB_reserved = s_df.sample(500, random_state=seed) if len(s_df) >= 500 else s_df.copy()

sA = s_df.drop(sB_reserved.index)

train_A_df = pd.concat([qA, sA]).drop_duplicates()

# ------------------------
# 3) Dataset B
#    = reserved 500 s_wrong_0
#    + remaining q_wrong_0 not in A
# ------------------------
qB = q_df.drop(qA.index)
sB = sB_reserved

train_B_df = pd.concat([qB, sB]).drop_duplicates()

# ------------------------
# Print stats
# ------------------------
print("Eval set:", len(eval_df))
print("Train A:", len(train_A_df))
print("Train B:", len(train_B_df))

print("\nOverlap checks:")
print("A ∩ B:", len(set(train_A_df.index) & set(train_B_df.index)))
print("Eval ∩ A:", len(set(eval_df.index) & set(train_A_df.index)))
print("Eval ∩ B:", len(set(eval_df.index) & set(train_B_df.index)))

# ------------------------
# Convert back to HF Datasets & save locally
# ------------------------
eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))
train_A_ds = Dataset.from_pandas(train_A_df.reset_index(drop=True))
train_B_ds = Dataset.from_pandas(train_B_df.reset_index(drop=True))

eval_ds.save_to_disk("prm_eval_500")
train_A_ds.save_to_disk("prm_train_A")
train_B_ds.save_to_disk("prm_train_B")

print("\n✅ Saved:")
print(" - prm_eval_500")
print(" - prm_train_A")
print(" - prm_train_B")
