import os
import glob
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from codecarbon import EmissionsTracker

# -------------------- FIXED / STANDARDIZED HYPERPARAMETERS --------------------
MODEL_NAME = "google/flan-t5-large"
PROMPT = (
    "classify the following text as sarcastic or not_sarcastic.\n"
    "Text: {text}\n"
    "Answer:"
)

PER_DEVICE_BATCH_SIZE = 4   # Stable for 6GB/12GB VRAM
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
MAX_INPUT_LENGTH = 512
MAX_LABEL_LENGTH = 32
SEED = 42
# -------------------------------------------------------------------------------

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def tokenize_batch(examples, tokenizer):
    label_texts = []
    for l in examples["label"]:
        l = str(l).strip().lower()
        if l in ["1", "sarcastic", "yes", "true"]:
            label_texts.append("sarcastic")
        else:
            label_texts.append("not_sarcastic")

    inputs = [PROMPT.format(text=t) for t in examples["text"]]
    
    # Do not pad here; let DataCollator handle it dynamically
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    labels = tokenizer(
        text_target=label_texts,
        truncation=True,
        max_length=MAX_LABEL_LENGTH,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_one_dataset(csv_path, outdir, epochs, learning_rate):
    # Define paths first to avoid NameError
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    adapter_out = os.path.join(outdir, f"{dataset_name}_adapter")
    os.makedirs(adapter_out, exist_ok=True)

    # Emissions directory for separate tracking
    emissions_dir = os.path.join(outdir, "emissions_reports")
    os.makedirs(emissions_dir, exist_ok=True)

    print(f"\n--- TRAINING START: {dataset_name} | epochs={epochs}, lr={learning_rate} ---")

    # Load dataset and split (60% training as requested)
    ds = load_dataset("csv", data_files=csv_path)["train"].shuffle(seed=SEED)
    train_ds = ds.select(range(int(0.6 * len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = train_ds.map(
        lambda x: tokenize_batch(x, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
    )

    # 8-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # Load Model in 8-bit
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Optimized for RTX 30-series stability
    )
    
    model.config.decoder_start_token_id = tokenizer.pad_token_id 
    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q", "k", "v", "o", "wi", "wo"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads() 

    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(adapter_out, "tmp"),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        bf16=True,      # Bfloat16 prevents the NaN gradients seen in your logs
        fp16=False,
        logging_steps=10,
        save_steps=500, # Save checkpoints for long runs
        save_total_limit=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, # Fixes the UserWarning
        optim="paged_adamw_8bit",

        report_to="none" # Prevents unwanted auto-logging to other platforms
    )

    # Data Collator (Handles label padding with -100 correctly)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer, # Updated from deprecated 'tokenizer'
        data_collator=data_collator,
    )

    # Initialize tracker for THIS specific execution
    tracker = EmissionsTracker(
        project_name=f"FineTune_{dataset_name}",
        output_dir=emissions_dir,
        output_file=f"{dataset_name}_emissions.csv",
        log_level="error" # Hides the messy 'INFO' heartbeat logs
    )

    tracker.start()
    try:
        trainer.train()
    finally:
        emissions_kg = tracker.stop()
        print(f"--- [COMPLETE] {dataset_name} CO2 footprint: {emissions_kg:.4f} kg ---")

    # Save final adapter
    model.save_pretrained(adapter_out)
    print(f"Adapter saved to {adapter_out}\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets_dir", required=True, help="Path to CSV file or folder")
    p.add_argument("--output_dir", default="./adapters", help="Base output directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)

    args = p.parse_args()
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle single file or directory
    if os.path.isdir(args.datasets_dir):
        csv_files = sorted(glob.glob(os.path.join(args.datasets_dir, "*.csv")))
    else:
        csv_files = [args.datasets_dir]

    for csv in csv_files:
        train_one_dataset(
            csv, 
            args.output_dir, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate
        )

if __name__ == "__main__":
    main()