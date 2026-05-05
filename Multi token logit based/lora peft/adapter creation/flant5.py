import random
import numpy as np
import pandas as pd
import torch
import argparse
import os
from tqdm.auto import tqdm

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from codecarbon import EmissionsTracker


# ========================= CONFIG =========================
MODEL_NAME = "google/flan-t5-large"

MAX_INPUT_LENGTH = 512
MAX_LABEL_LENGTH = 32
BATCH_SIZE = 2
EPOCHS = 4
LEARNING_RATE = 5e-5
SEED = 42

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
# ==========================================================


# ----------------- ARGUMENTS -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--lr", type=float, default=LEARNING_RATE)
args = parser.parse_args()

CSV_PATH = args.csv
OUTPUT_DIR = args.output_dir
EPOCHS = args.epochs
LEARNING_RATE = args.lr


# ----------------- SEED -----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ----------------- PROMPT -----------------
def make_prompt(text):
    return (
        "Classify the following text as sarcastic or not sarcastic.\n"
        f'Text: "{text}"\n'
        "Answer: "
    )


# ----------------- LOAD DATA -----------------
df = pd.read_csv(CSV_PATH)
dataset = Dataset.from_pandas(df).shuffle(seed=SEED)

split = dataset.train_test_split(test_size=0.4, seed=SEED)
train_ds = split["train"]

# Save test split
os.makedirs(OUTPUT_DIR, exist_ok=True)
test_csv_path = os.path.join(OUTPUT_DIR, "test_split_40.csv")
split["test"].to_pandas().to_csv(test_csv_path, index=False)
print(f"✅ Test split saved to {test_csv_path}")


# ----------------- TOKENIZER -----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ----------------- MODEL (8-bit) -----------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# IMPORTANT FIX
model.config.decoder_start_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)


# ----------------- LORA -----------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    target_modules=["q", "k", "v", "o"],   # as you requested
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ----------------- LOGIT FUNCTION -----------------
def compute_logprob_batch(prompts, label_texts):

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(device)

    labels = tokenizer(
        label_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LABEL_LENGTH
    ).input_ids.to(device)

    # ignore padding tokens
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        labels=labels
    )

    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    scores = []

    for i in range(labels.size(0)):
        score = torch.tensor(0.0, device=device)
        count = 0

        for t in range(labels.size(1)):
            token_id = labels[i, t].item()

            if token_id == -100:
                continue

            score += log_probs[i, t, token_id]
            count += 1

        score = score / max(count, 1)
        scores.append(score)

    return torch.stack(scores)


# ----------------- LOSS FUNCTION -----------------
def compute_margin_loss(texts, labels):

    prompts = [make_prompt(t) for t in texts]

    correct = [
        " sarcastic" if str(l).lower() in ["1", "sarcastic"] else " not sarcastic"
        for l in labels
    ]

    wrong = [
        " not sarcastic" if c == " sarcastic" else " sarcastic"
        for c in correct
    ]

    correct_scores = compute_logprob_batch(prompts, correct)
    wrong_scores = compute_logprob_batch(prompts, wrong)

    margin = correct_scores - wrong_scores

    loss = -torch.log(torch.sigmoid(margin)).mean()

    return loss


# ----------------- TRAINING LOOP -----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()

tracker = EmissionsTracker(
    project_name="FLAN-T5-LoRA-Logit-Training",
    output_dir=OUTPUT_DIR,
    output_file="train_emissions.csv",
    log_level="error"
)
tracker.start()

print("\n🚀 Starting Logit-based LoRA Training...\n")

train_df = train_ds.to_pandas()

for epoch in range(EPOCHS):

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    total_loss = 0

    for i in tqdm(range(0, len(train_df), BATCH_SIZE)):

        batch = train_df.iloc[i:i+BATCH_SIZE]

        texts = batch["text"].tolist()
        labels = batch["label"].tolist()

        loss = compute_margin_loss(texts, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / (len(train_df) // BATCH_SIZE)

    print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

emissions = tracker.stop()
print(f"\n🌱 CO2 emissions (kg): {emissions:.6f}")


# ----------------- SAVE -----------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ FLAN-T5 LoRA training completed!")