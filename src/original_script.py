# ============================================================
# Fine-tuning BERTimbau (Português) para Classificação de Texto
# ============================================================
import os, math, random, inspect
import numpy as np
import pandas as pd
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ------------------------------------------------------------
# Imports principais
# ------------------------------------------------------------
import transformers
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer, TrainingArguments
)
print("transformers:", transformers.__version__)

# Helper: TrainingArguments compatível com a sua versão
def build_training_arguments(**kwargs) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys()); allowed.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = [k for k in kwargs if k not in allowed]
    if dropped:
        print("[Aviso] parâmetros ignorados nesta versão:", dropped)
    return TrainingArguments(**filtered)

# ------------------------------------------------------------
# 1) FONTE DOS DADOS (escolha UMA)
# ------------------------------------------------------------
# (A) Dataset Hugging Face (deve ter colunas 'text' e 'label')
HF_DATASET = None
HF_CONFIG  = None  # ex.: "default" ou subtarefa (se houver)

# (B) CSVs locais com colunas: text,label (rótulo pode ser string)
CSV_TRAIN = None  # ex.: "/content/train.csv"
CSV_VAL   = None  # ex.: "/content/val.csv"

# (C) Fallback didático embutido (PT-BR)
FALLBACK_PT = [
    ("o filme é excelente, emocionante e muito bem dirigido.", 1),
    ("péssimo atendimento, não volto mais.", 0),
    ("a comida estava maravilhosa, sabores incríveis.", 1),
    ("produto chegou quebrado e atrasado, experiência horrível.", 0),
    ("serviço rápido e eficiente, gostei bastante.", 1),
    ("interface confusa e cheia de bugs.", 0),
    ("uma experiência fantástica do começo ao fim!", 1),
    ("não recomendo, custo-benefício muito ruim.", 0),
]

# ------------------------------------------------------------
# 2) Carregar dados
# ------------------------------------------------------------
train_texts, train_labels = [], []
val_texts,   val_labels   = [], []

def load_from_hf(name, config=None):
    from datasets import load_dataset
    ds = load_dataset(name, config) if config else load_dataset(name)
    # tenta achar colunas text/label comuns
    # você pode adaptar aqui se seu dataset tiver outros nomes
    def pick_cols(split):
        cand_text = [c for c in ["text", "sentence", "texto", "review", "content"] if c in ds[split].column_names]
        cand_label= [c for c in ["label", "labels", "sentiment", "classe"] if c in ds[split].column_names]
        assert cand_text and cand_label, f"Não encontrei colunas 'text' e 'label' no split {split}. Colunas: {ds[split].column_names}"
        return cand_text[0], cand_label[0]

    tcol_tr, lcol_tr = pick_cols("train")
    tcol_va, lcol_va = pick_cols("validation") if "validation" in ds else pick_cols("test")

    Xtr = ds["train"][tcol_tr];  Ytr = ds["train"][lcol_tr]
    Xva = ds["validation"][tcol_va] if "validation" in ds else ds["test"][tcol_va]
    Yva = ds["validation"][lcol_va] if "validation" in ds else ds["test"][lcol_va]
    return list(Xtr), list(Ytr), list(Xva), list(Yva)

def load_from_csv(path):
    df = pd.read_csv(path)
    assert "text" in df.columns and "label" in df.columns, f"O CSV {path} deve ter colunas: text,label"
    return df["text"].tolist(), df["label"].tolist()

try:
    if HF_DATASET:
        print(f"Carregando dataset HF: {HF_DATASET} ({HF_CONFIG})")
        train_texts, train_labels, val_texts, val_labels = load_from_hf(HF_DATASET, HF_CONFIG)
    elif CSV_TRAIN and CSV_VAL:
        print("Carregando CSVs locais…")
        train_texts, train_labels = load_from_csv(CSV_TRAIN)
        val_texts,   val_labels   = load_from_csv(CSV_VAL)
    else:
        print("Usando fallback didático embutido (PT-BR).")
        pairs = FALLBACK_PT[:]
        random.shuffle(pairs)
        # split 75/25
        n = int(0.75 * len(pairs))
        tr, va = pairs[:n], pairs[n:]
        train_texts = [t for t, y in tr]; train_labels = [y for t, y in tr]
        val_texts   = [t for t, y in va]; val_labels   = [y for t, y in va]
except Exception as e:
    raise RuntimeError(f"Falha ao carregar dados: {e}")

print(f"Tamanho: train={len(train_texts)}  val={len(val_texts)}")

# ------------------------------------------------------------
# 3) Saneamento (garante list[str], remove NaN/None/bytes) + map labels
# ------------------------------------------------------------
def _is_nan(x):
    try: return bool(np.isnan(x))
    except Exception: return False

def to_str(x):
    if x is None: return None
    if isinstance(x, (bytes, bytearray)):
        try: x = x.decode("utf-8", "ignore")
        except Exception: x = str(x)
    if isinstance(x, (np.generic,)): x = x.item()
    if isinstance(x, (float, np.floating)) and _is_nan(x): return None
    s = str(x).strip()
    return s if s else None

def clean_xy(X, y, name="split"):
    Xo, yo = [], []
    bad = 0
    for t, l in zip(X, y):
        s = to_str(t)
        if s is None: bad += 1; continue
        Xo.append(s)
        yo.append(l)
    if bad: print(f"[{name}] {bad} amostras removidas por texto inválido.")
    return Xo, yo

train_texts, train_labels = clean_xy(train_texts, train_labels, "train")
val_texts,   val_labels   = clean_xy(val_texts,   val_labels,   "val")

# Mapear labels (strings → ids)
uniq = sorted({str(l) for l in (list(train_labels) + list(val_labels))})
label2id = {lab:i for i, lab in enumerate(uniq)}
id2label = {i:lab for lab,i in label2id.items()}
train_labels = [label2id[str(l)] for l in train_labels]
val_labels   = [label2id[str(l)] for l in val_labels]
num_labels = len(label2id)
print("Labels:", label2id)

# ------------------------------------------------------------
# 4) Tokenizer (BERTimbau) e tokenização
# ------------------------------------------------------------
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"   # ou "…-uncased"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

MAX_LEN = 160
def tokenize_batch(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

train_enc = tokenize_batch(train_texts)
val_enc   = tokenize_batch(val_texts)

class TorchTextDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = TorchTextDataset(train_enc, train_labels)
val_ds   = TorchTextDataset(val_enc,   val_labels)

# ------------------------------------------------------------
# 5) Modelo e (opcional) congelamento parcial do encoder
# ------------------------------------------------------------
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

FREEZE_N_LAYERS = 0  # ex.: 6 para congelar 6 camadas iniciais
if FREEZE_N_LAYERS > 0:
    # Congelar embeddings + primeiras N camadas do encoder
    for p in model.bert.embeddings.parameters():
        p.requires_grad = False
    for i in range(FREEZE_N_LAYERS):
        for p in model.bert.encoder.layer[i].parameters():
            p.requires_grad = False
    print(f"Camadas congeladas: embeddings + {FREEZE_N_LAYERS} primeiras camadas.")

# ------------------------------------------------------------
# 6) Métricas (accuracy + F1 se disponível)
# ------------------------------------------------------------
try:
    import evaluate
    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        r1 = acc_metric.compute(predictions=preds, references=labels)
        r2 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        return {"accuracy": r1["accuracy"], "f1": r2["f1"]}
except Exception:
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

# ------------------------------------------------------------
# 7) Treinamento (Trainer)
# ------------------------------------------------------------
EPOCHS = 3 if len(train_texts) >= 1000 else 5
BATCH  = 16 if torch.cuda.is_available() else 8

args = build_training_arguments(
    output_dir="bertimbau-cls-ptbr",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("\n=== Iniciando fine-tuning do BERTimbau ===")
trainer.train()
eval_out = trainer.evaluate()
print("\nResultados de validação:", eval_out)

# ------------------------------------------------------------
# 8) Inferência em frases PT-BR
# ------------------------------------------------------------
def predict(texts):
    model.eval()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
    decoded = [(t, id2label[int(p)], probs[i]) for i,(t,p) in enumerate(zip(texts, preds))]
    return decoded

amostras = [
    "o atendimento foi excelente e rápido.",
    "que decepção, não recomendo a ninguém.",
    "funciona bem, mas poderia ser mais intuitivo."
]
for texto, pred, prob in predict(amostras):
    print(f"- {texto}\n  -> classe: {pred} | probs={np.round(prob, 3)}")