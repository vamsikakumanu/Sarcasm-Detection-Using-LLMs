import random
import numpy as np
import pandas as pd
import torch
import argparse
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from codecarbon import EmissionsTracker

# ========================= CONFIG =========================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

CSV_PATH = "E:\\preprocessed datasets\\IACV1.csv"   # id,text,label
OUTPUT_DIR = r"F:\lora\adapters\qwen2_5_0_5B\IACV1"

PROMPT = (
    "Classify the following text as sarcastic or not_sarcastic\n"
    "Text: {text}\n"
    "Answer:"
)

MAX_INPUT_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
EPOCHS = 4
LEARNING_RATE = 5e-5
SEED = 42

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
# ==========================================================


# ----------------- ARGUMENTS -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default=CSV_PATH)
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--lr", type=float, default=LEARNING_RATE)
args = parser.parse_args()

CSV_PATH = args.csv
OUTPUT_DIR = args.output_dir
EPOCHS = args.epochs
LEARNING_RATE = args.lr
# --------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ----------------- LOAD DATA -----------------
df = pd.read_csv(CSV_PATH)
dataset = Dataset.from_pandas(df).shuffle(seed=SEED)

split = dataset.train_test_split(test_size=0.4, seed=SEED)
train_ds = split["train"]   # 60%
test_ds = split["test"]     # 40%

test_csv_path = "test_split_40.csv"
test_ds.to_pandas().to_csv(test_csv_path, index=False)
print(f"✅ Test split saved to {test_csv_path}")


# ----------------- PROMPT BUILD -----------------
def build_example(row):
    prompt = PROMPT.format(text=row["text"])
    return {"text": f"{prompt} {row['label']}"}

train_ds = train_ds.map(
    build_example,
    remove_columns=train_ds.column_names
)


# ----------------- TOKENIZER -----------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(example):
    out = tokenizer(
        example["text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_ds = train_ds.map(tokenize, batched=False)


# ----------------- MODEL (8-bit) -----------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)


# ----------------- LORA (QWEN-2.5 FIX) -----------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ----------------- TRAINING -----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    ),
)
energy_dir = os.path.join(OUTPUT_DIR, "energy")
os.makedirs(energy_dir, exist_ok=True)
tracker = EmissionsTracker(
    project_name="Qwen2.5-LoRA-Training",
    experiment_id="IACV1_60pct",
    output_dir=os.path.join(OUTPUT_DIR, "energy"),
    output_file="train_emissions.csv",
    log_level="error"
)
tracker.start()

trainer.train()
emissions = tracker.stop()
print(f"\nTotal CO2 emissions (kg): {emissions:.6f}")

# ----------------- SAVE -----------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Qwen-2.5-0.5B LoRA adapter trained successfully")
