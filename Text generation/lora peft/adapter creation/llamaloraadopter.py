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

# ---------------- CONFIG (defaults) ----------------
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CSV_PATH = "E:\\preprocessed datasets\\IACV1.csv"      # id,text,label
OUTPUT_DIR = r"F:\lora\adapters\llamaadopter\IACV1"

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
# --------------------------------------------------

# -------- Command-line arguments (ONLY requested ones) --------
parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--lr", type=float, default=LEARNING_RATE)
parser.add_argument("--csv", type=str, default=CSV_PATH)
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

args = parser.parse_args()

EPOCHS = args.epochs
LEARNING_RATE = args.lr
CSV_PATH = args.csv
OUTPUT_DIR = args.output_dir
# --------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -------- Load & split dataset (60:40) --------
df = pd.read_csv(CSV_PATH)
dataset = Dataset.from_pandas(df).shuffle(seed=SEED)

split = dataset.train_test_split(test_size=0.4, seed=SEED)
train_ds = split["train"]     # 60%
test_ds = split["test"]       # 40%

# Save test split for Code-2

test_csv_path = os.path.join(OUTPUT_DIR, "test_split_40.csv")
test_ds.to_pandas().to_csv(test_csv_path, index=False)
print(f"✅ Test split saved to {test_csv_path}")

# -------- Build training prompt --------
def build_example(row):
    prompt = PROMPT.format(text=row["text"])
    return {"text": f"{prompt} {row['label']}"}

train_ds = train_ds.map(
    build_example,
    remove_columns=train_ds.column_names
)

# -------- Tokenizer --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = train_ds.map(tokenize)

# -------- 8-bit base model (Pure LoRA) --------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# -------- LoRA config --------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------- Training --------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
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
    project_name="Llama3.2-LoRA-Training",
    experiment_id="IACV1_60pct",
    output_dir=os.path.join(OUTPUT_DIR, "energy"),
    output_file="train_emissions.csv",
    log_level="error"
)
tracker.start()
trainer.train()
emissions = tracker.stop()

print("\n================ ENERGY & EMISSIONS =================")
print(f"Total CO2 emissions (kg): {emissions:.6f}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Adapter trained on 60% of the data.")
