#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Student-only PRM training (NO teacher)
Loss: Bradley–Terry ONLY

- Model: Qwen/Qwen2.5-Math-1.5B + added </think> + PRM head (2 logits)
- Pool at last </think>
- 2 positives: sol_with_think, sft_solution
- 1–2 wrongs: build exactly 2 BT pairs per row as described
- Train schedule:
    2 epochs on prm_train_A/
    eval
    5 epochs on prm_train_B/ with eval after each
"""

import os, math, random
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm
from modeling_student_prm import StudentPRM, StudentPRMConfig
import shutil

SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Provide step-by-step reasoning and put the final answer in \\boxed{}."

# ---------------- Config ----------------
SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

BATCH_SIZE = 2
LR = 1e-5
WEIGHT_DECAY = 0.01


STUDENT_BASE = "Qwen/Qwen2.5-Math-1.5B"
STUDENT_POOL_TOKEN = "</think>"

PATH_TRAIN_A = "prm_train_A"
PATH_TRAIN_B = "prm_train_B"
PATH_EVAL    = "prm_eval_500"

SAVE_DIR = "qwen2p5_math_1p5B_prm_student_no_teacher"
HF_REPO_ID = "omrisap/Qwen2.5-Math-PRM-1.5B"

PUSH_TO_HUB = True  # set HF_TOKEN env var outside

MAX_LEN = 4096


# -------------- Utils ------------------
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def non_empty(x):
    if x is None: return False
    if isinstance(x, float) and math.isnan(x): return False
    if isinstance(x, str) and x.strip()=="": return False
    return True

def build_chat(tokenizer, question, solution, end_token):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
        {"role": "assistant", "content": solution + end_token},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True)

def last_token_index(input_ids, token_id):
    mask = (input_ids == token_id)
    flipped = torch.flip(mask, dims=[1])
    idx_from_end = torch.argmax(flipped.int(), dim=1)
    return (input_ids.shape[1]-1) - idx_from_end


# -------------- Student PRM Model ------------------
# Removed local StudentPRM definition; using modeling_student_prm.StudentPRM instead.


# -------------- Dataset -> BT pairs ------------------
@dataclass
class Pair:
    q: str
    pos: str
    neg: str

class PairDataset(Dataset):
    def __init__(self, hf_split, seed=SEED):
        self.rows=[]
        random.seed(seed)
        for r in hf_split:
            q = r["problem"]
            p1, p2 = r.get("sol_with_think"), r.get("sft_solution")
            if not (non_empty(p1) and non_empty(p2)): continue
            wrongs = [r.get(k) for k in ["q_wrong_0","q_wrong_1","s_wrong_0","s_wrong_1"] if non_empty(r.get(k))]
            if len(wrongs)==0: continue
            if len(wrongs)>=2:
                w1, w2 = wrongs[0], wrongs[1]
                pos_list=[p1,p2]
                self.rows.append(Pair(q, random.choice(pos_list), w1))
                self.rows.append(Pair(q, random.choice(pos_list), w2))
            else:
                w = wrongs[0]
                self.rows.append(Pair(q, p1, w))
                self.rows.append(Pair(q, p2, w))

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


# -------------- Collator ------------------
class Collator:
    def __init__(self, tok, maxlen):
        self.tok = tok; self.maxlen = maxlen

    def __call__(self, batch):
        texts=[]
        for b in batch:
            texts.append(build_chat(self.tok, b.q, b.pos, STUDENT_POOL_TOKEN))
            texts.append(build_chat(self.tok, b.q, b.neg, STUDENT_POOL_TOKEN))
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.maxlen)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask")
        }


# -------------- BT loss ------------------
def bt_loss(logits):
    pos = logits[0::2,1]  # assign positive class logit
    neg = logits[1::2,1]
    return -torch.log(torch.sigmoid(pos-neg)+1e-12).mean()


# -------------- Eval ------------------
@torch.no_grad()
def evaluate(eval_split, tok, model):
    model.eval()
    pool_id = tok.convert_tokens_to_ids(STUDENT_POOL_TOKEN)
    model.pool_id = pool_id
    wrong_cols=["q_wrong_0","q_wrong_1","s_wrong_0","s_wrong_1"]

    def score(q, sols):
        if not sols: return []
        chats=[build_chat(tok,q,s,STUDENT_POOL_TOKEN) for s in sols]
        print(f'Eval struxture is {chats[0]}')
        enc = tok(chats, return_tensors="pt", padding=True, truncation=True,max_length=MAX_LEN).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            logits = model(enc["input_ids"],enc.get("attention_mask")).logits  # updated for SequenceClassifierOutput
        return logits[:,1].float().cpu().tolist()

    correct = ["sol_with_think","sft_solution"]
    totals={}; counts={}
    for c in correct:
        for w in wrong_cols:
            totals[(c,w)]=0; counts[(c,w)]=0

    for r in tqdm(eval_split, desc="Eval", leave=False):
        if not(non_empty(r.get("sol_with_think")) and non_empty(r.get("sft_solution"))): continue
        q=r["problem"]
        pA,pB = r["sol_with_think"], r["sft_solution"]
        ps = score(q,[pA,pB])
        if len(ps)<2: continue
        for w in wrong_cols:
            wr=r.get(w)
            if not non_empty(wr): continue
            ws = score(q,[wr])[0]
            counts[("sol_with_think",w)] +=1; totals[("sol_with_think",w)] += (ps[0]>ws)
            counts[("sft_solution",w)]   +=1; totals[("sft_solution",w)]   += (ps[1]>ws)

    result={}
    for k in totals:
        if counts[k]>0:
            result[k]=totals[k]/counts[k]
    return result


# -------------- Training ------------------
def train_loop():
    set_seed(SEED)

    trainA = load_from_disk(PATH_TRAIN_A).select(range(5))

    trainB = load_from_disk(PATH_TRAIN_B)
    evalS  = load_from_disk(PATH_EVAL)


    tok = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)
    if STUDENT_POOL_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens":[STUDENT_POOL_TOKEN]})

    base = AutoModel.from_pretrained(STUDENT_BASE, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True)
    base.resize_token_embeddings(len(tok))
    config = StudentPRMConfig(base_model_name=STUDENT_BASE, pool_token=STUDENT_POOL_TOKEN, hidden_size=base.config.hidden_size, vocab_size=len(tok), num_labels=2)
    model = StudentPRM(config, base=base, tokenizer=tok).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dsA = PairDataset(trainA); dlA = DataLoader(dsA, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collator(tok,MAX_LEN))
    dsB = PairDataset(trainB); dlB = DataLoader(dsB, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collator(tok,MAX_LEN))

    total_steps = 1*len(dlA) + 1*len(dlB)  # reflect actual planned epochs
    sched = get_cosine_schedule_with_warmup(optimizer, int(0.05*total_steps), total_steps)

    def run_epoch(loader,name):
        model.train()
        pbar=tqdm(loader,desc=name)
        avg=0; n=0
        for batch in pbar:
            with torch.autocast(device_type="cuda",dtype=DTYPE):
                print(f'full text of first question {tok.decode(batch["input_ids"][0])}')
                logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)).logits  # updated
                loss = bt_loss(logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); sched.step(); optimizer.zero_grad()
            avg+=loss.item(); n+=1
            pbar.set_postfix({"L_BT":f"{avg/n:.4f}"})


    for e in range(1):
        run_epoch(dlA,f"A-{e+1}")

    # eval
    r=evaluate(evalS,tok,model)
    print("\n[Eval after 2 epochs A]")
    for k,v in r.items(): print(k,":",f"{v:.4f}")


    for e in range(1):
        run_epoch(dlB,f"B-{e+1}")
        r=evaluate(evalS,tok,model)
        print(f"\n[Eval after B-{e+1}]")
        for k,v in r.items(): print(k,":",f"{v:.4f}")

    # save (HF compatible)
    os.makedirs(SAVE_DIR,exist_ok=True)
    tok.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)
    # ensure modeling file present for remote code auto_map
    shutil.copyfile("modeling_student_prm.py", f"{SAVE_DIR}/modeling_student_prm.py")
    # minimal README for hub
    with open(f"{SAVE_DIR}/README.md","w") as f:
        f.write(f"""# Qwen2.5 Math PRM Student (1.5B)\n\nCustom pairwise reward model head (2 logits) on top of {STUDENT_BASE}.\nPool at last occurrence of token `{STUDENT_POOL_TOKEN}`.\n\n## Usage\n```python\nfrom transformers import AutoTokenizer, AutoModel\nrepo = '{HF_REPO_ID}'\ntok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)\nmodel = AutoModel.from_pretrained(repo, trust_remote_code=True)\n# logits shape: (batch, 2)\n```\n""")

    print(f"\n✅ Saved HF format to {SAVE_DIR}")

    if PUSH_TO_HUB:
        from huggingface_hub import HfApi, login
        token = os.getenv("HF_TOKEN", HF_TOKEN)
        if not token:
            print("⚠️ HF_TOKEN not provided; skipping hub push.")
        else:
            login(token)
            api = HfApi()
            api.create_repo(HF_REPO_ID, private=False, exist_ok=True)
            api.upload_folder(repo_id=HF_REPO_ID, folder_path=SAVE_DIR)
            print(f"✅ Pushed to https://huggingface.co/{HF_REPO_ID}")
    elif PUSH_TO_HUB and not hf_token:
        print("⚠️ HF_TOKEN env var not set; skipping hub push.")


if __name__=="__main__":
    train_loop()
