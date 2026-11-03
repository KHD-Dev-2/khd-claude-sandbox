import json

import math

import re

import os

import random

import hashlib

import subprocess

import sys

from pathlib import Path



import torch

import torch.nn as nn

import torch.nn.functional as F



subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=False)

from tokenizers import Tokenizer

from tokenizers.models import WordLevel

from tokenizers.pre_tokenizers import Whitespace

from tokenizers.trainers import WordLevelTrainer



INPUT = Path("/kaggle/input")

OUT = Path("/kaggle/working/khd-gen4")

OUT.mkdir(parents=True, exist_ok=True)

RESULT = Path("/kaggle/working/khd-gen4-result.json")

torch.manual_seed(43)

random.seed(43)



train_path = sorted(INPUT.rglob("train.txt"))[0]
val_path = sorted(INPUT.rglob("val.txt"))[0]
holdout_path = sorted(INPUT.rglob("holdout24.txt"))[0]
# Preprocessing contract, recorded in config so eval_gen11.py applies the same to prompts (it used to guess from the model name).
PREPROCESS = {"digits": True, "sequences": True}
def digitize(t): return re.sub(r"[0-9]+(?:[.,][0-9]+)?", lambda m: " ".join(m.group(0).replace(",", " , ").replace(".", " . ")), t)
def seqsplit(t): return re.sub(r"\b[ACGTU]{2,}\b", lambda m: " ".join(m.group(0)), t)
def prep(t):
    if PREPROCESS["digits"]: t = digitize(t)
    if PREPROCESS["sequences"]: t = seqsplit(t)
    return t
train_lines = [prep(x.strip()) for x in re.split(r"\n\s*\n", train_path.read_text(encoding="utf-8")) if x.strip()]
val_lines = [prep(x.strip()) for x in re.split(r"\n\s*\n", val_path.read_text(encoding="utf-8")) if x.strip()]
holdout_lines = [prep(x.strip()) for x in re.split(r"\n\s*\n", holdout_path.read_text(encoding="utf-8")) if x.strip()]
tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(train_lines, WordLevelTrainer(vocab_size=4096, min_frequency=1, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))
vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
vocab = [token for token, _ in vocab]
stoi = {token: i for i, token in enumerate(vocab)}
pad_id = stoi["<pad>"]; bos_id = stoi["<bos>"]; eos_id = stoi["<eos>"]; unk_id = stoi["<unk>"]
config = {"vocab_size": len(vocab), "context": 256, "d_model": 512, "layers": 1, "loops": 6, "heads": 8, "d_ff": 2048, "dropout": 0.1, "tokenizer": "WordLevel/Whitespace", "vocab_size_requested": 4096, "preprocess": PREPROCESS}
ctx = config["context"]
def make_example(line):
    if "Réponse :" not in line: return None
    before, after = line.split("Réponse :", 1)
    before = before.replace("Question :", "", 1).strip()
    qids = tokenizer.encode("Question : " + before + " Réponse :").ids
    aids = tokenizer.encode(after.strip()).ids + [eos_id]
    ids = [bos_id] + qids + aids
    labels = [-100] * (1 + len(qids)) + aids
    # The position table has ctx rows; ctx + 1 tokens would index past it.
    if len(ids) > ctx:
        ids = ids[:ctx]; labels = labels[:ctx]
    return ids, labels
train_examples = [e for e in (make_example(x) for x in train_lines) if e]
val_examples = [e for e in (make_example(x) for x in val_lines) if e]
answer_chars = sum(max(1, len(x.split("Réponse :",1)[1])) for x in train_lines if "Réponse :" in x)

class Block(nn.Module):

    def __init__(self, d, heads, ff):

        super().__init__()

        self.ln1 = nn.LayerNorm(d)

        self.qkv = nn.Linear(d, 3 * d, bias=False)

        self.proj = nn.Linear(d, d, bias=False)

        self.ln2 = nn.LayerNorm(d)

        self.fc1 = nn.Linear(d, ff)

        self.fc2 = nn.Linear(ff, d)

        self.heads = heads

        self.drop = nn.Dropout(0.1)

        self.d = d

    def forward(self, x):

        b, t, d = x.shape

        h = self.ln1(x)

        q, k, v = self.qkv(h).chunk(3, dim=-1)

        q = q.view(b, t, self.heads, d // self.heads).transpose(1, 2)

        k = k.view(b, t, self.heads, d // self.heads).transpose(1, 2)

        v = v.view(b, t, self.heads, d // self.heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // self.heads)

        mask = torch.tril(torch.ones(t, t, device=x.device)).view(1, 1, t, t)

        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(b, t, d)

        x = x + self.drop(self.proj(out))

        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))

        return x



class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.wpe = nn.Embedding(cfg["context"], cfg["d_model"])
        self.block = Block(cfg["d_model"], cfg["heads"], cfg["d_ff"])
        self.loop_emb = nn.Embedding(cfg["loops"], cfg["d_model"])
        self.lnf = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.context = cfg["context"]; self.loops = cfg["loops"]
    def forward(self, idx, targets=None):
        b,t = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        for loop in range(self.loops):
            step = torch.full((b,t), loop, dtype=torch.long, device=idx.device)
            x = self.block(x + self.loop_emb(step))
        logits = self.head(self.lnf(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8):
        for _ in range(n):
            logits, _ = self(idx[:, -self.context:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyGPT(config).to(device)

def make_batch(examples, batch_size):

    chosen = [random.choice(examples) for _ in range(batch_size)]

    maxlen = max(len(x[0]) for x in chosen)

    xb = torch.full((batch_size, maxlen), pad_id, dtype=torch.long, device=device)

    yb = torch.full((batch_size, maxlen), -100, dtype=torch.long, device=device)

    for i, (ids, labels) in enumerate(chosen):

        xb[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        yb[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)

    return xb, yb

def masked_loss(xb, yb):

    logits, _ = model(xb)

    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), yb[:, 1:].reshape(-1), ignore_index=-100)

with torch.no_grad():

    xb0, yb0 = make_batch(train_examples, 1)

    initial_loss = float(masked_loss(xb0, yb0).item())

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)

steps = 4000; losses = []

for step in range(1, steps + 1):

    xb, yb = make_batch(train_examples, 16)

    loss = masked_loss(xb, yb)

    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

    if step % 50 == 0 or step == 1:

        losses.append({"step": step, "loss": round(float(loss.item()), 5)}); print(losses[-1])

model.eval()

val_losses = []

with torch.no_grad():

    for _ in range(20):

        xb, yb = make_batch(val_examples, 12)

        val_losses.append(float(masked_loss(xb, yb).item()))

val_loss = sum(val_losses) / len(val_losses)

train_answer_tokens = sum(sum(1 for l in labels if l != -100) for _, labels in train_examples)

tokens_per_char = train_answer_tokens / max(1, answer_chars)

bits_per_char = (val_loss / math.log(2)) * tokens_per_char

prompt = "Question : Dérive f(x)=x^3.\nRéponse :"

prompt_ids = [bos_id] + tokenizer.encode(prompt).ids

idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

generated = model.generate(idx, 40, 0.7)[0].tolist()

sample = tokenizer.decode(generated, skip_special_tokens=True)

holdout_answers = []
for line in holdout_lines:
    prompt_ids = [bos_id] + tokenizer.encode(line).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(idx, 35, 0.7)[0].tolist()
    holdout_answers.append(tokenizer.decode(generated, skip_special_tokens=True))

torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "step": steps, "initial_loss": initial_loss}, OUT / "checkpoint.pt")

weights = {}

for name, tensor in model.state_dict().items():

    if tensor.dim() > 1:

        weights[name] = [[round(float(v), 6) for v in row] for row in tensor.detach().cpu().reshape(tensor.shape[0], -1).tolist()]

    else:

        weights[name] = [round(float(v), 6) for v in tensor.detach().cpu().tolist()]

(OUT / "weights.json").write_text(json.dumps(weights, separators=(",", ":")), encoding="utf-8")

(OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

(OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

(OUT / "tokenizer.json").write_bytes(tokenizer.to_str().encode())

(OUT / "README.md").write_text("# KHD Gen11 recurrent\n\nModÃ¨le expÃ©rimental initialisÃ© par KHD, tokenizer WordLevel entraÃ®nÃ© sur le corpus KHD. Pas gÃ©nÃ©raliste.\n", encoding="utf-8")

result = {"status": "COMPLETE", "params": sum(p.numel() for p in model.parameters()), "vocab_size": len(vocab),

    "train_paragraphs": len(train_lines), "val_paragraphs": len(val_lines), "answer_tokens": train_answer_tokens, "initial_loss": round(initial_loss, 5), "checkpoint_bytes": (OUT / "checkpoint.pt").stat().st_size,

    "steps": steps, "final_loss": losses[-1]["loss"] if losses else None, "val_loss": round(val_loss, 5),

    "bits_per_char": round(bits_per_char, 5), "sample_prompt": prompt, "sample": sample, "holdout_answers": holdout_answers,

    "weights_sha256": hashlib.sha256((OUT / "weights.json").read_bytes()).hexdigest(),

    "weights_bytes": (OUT / "weights.json").stat().st_size,

    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,

    "note": "KHD Gen11 recurrent fully neural, trained from scratch on 10,817 verified science examples; no tool fallback."}

RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print("KHD_GEN2_RESULT=" + json.dumps(result, ensure_ascii=False)[:4000])

