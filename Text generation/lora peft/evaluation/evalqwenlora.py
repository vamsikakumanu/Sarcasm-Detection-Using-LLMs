import os
import re
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from codecarbon import EmissionsTracker

from tqdm.auto import tqdm

# ================= CONFIG =================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = (
    "Classify the following text as sarcastic or not_sarcastic\n"
    "Text: {text}\n"
    "Answer:"
)

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 5
BATCH_SIZE = 4
SEED = 42
# ==========================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_prediction(text: str) -> str:
    if not text:
        return "unknown"

    s = text.lower().strip()
    s = re.sub(r'(answer:|classification:)', '', s).strip()

    if s.startswith("not_sarcastic") or s.startswith("not sarcastic") or s.startswith("not"):
        return "not_sarcastic"
    if s.startswith("sarcastic") or s.startswith("sarc"):
        return "sarcastic"

    return "unknown"


def main(args):
    set_seed(SEED)
        # ================= ENERGY & EMISSIONS TRACKING (LoRA Inference) =================
    tracker = EmissionsTracker(
        project_name="Qwen2.5-LoRA-Inference",
        experiment_id="lora_adapter_testing",
        output_dir=args.output_dir,
        output_file="lora_test_emissions.csv",
        log_level="error"
    )
    tracker.start()
    # ===============================================================================


    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    df = pd.read_csv(args.test_csv)
    print(f"Loaded test samples: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(
         MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    gold_labels, pred_labels, raw_generations = [], [], []

    print("\nStarting evaluation...\n")

    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[start:start + BATCH_SIZE]

        texts = batch["text"].tolist()
        golds = batch["label"].tolist()

        prompts = [
    PROMPT.format(text=t)
    for t in texts
]


        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # ✅ CRITICAL FIX: decode ONLY generated tokens
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

        decoded_batch = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        for i, gen in enumerate(decoded_batch):
            raw_answer = gen.strip()
            pred = normalize_prediction(raw_answer)

            gold_labels.append(golds[i])
            pred_labels.append(pred)
            raw_generations.append(raw_answer)

            print(f"\n[{start + i}]")
            print(f"TEXT: {texts[i]}")
            print(f"GOLD: {golds[i]}")
            print(f"RAW_GEN: {raw_answer}")
            print(f"PRED: {pred}")

    # -------- Metrics --------
    filtered_gold = [g for g, p in zip(gold_labels, pred_labels) if p != "unknown"]
    filtered_pred = [p for p in pred_labels if p != "unknown"]

    acc = accuracy_score(filtered_gold, filtered_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_gold,
        filtered_pred,
        average="binary",
        pos_label="sarcastic",
        zero_division=0,
    )

    mcc = matthews_corrcoef(filtered_gold, filtered_pred)

    cm = confusion_matrix(filtered_gold, filtered_pred, labels=["sarcastic", "not_sarcastic"])

    report = classification_report(filtered_gold, filtered_pred, digits=4)

    print("\n================ METRICS ================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(cm)
    print(report)

    os.makedirs(args.output_dir, exist_ok=True)

    df["prediction"] = pred_labels
    df["raw_generation"] = raw_generations
    df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    print(f"\n✅ Results saved to: {args.output_dir}")
        # ================= ENERGY & EMISSIONS RESULTS =================
    emissions = tracker.stop()
    print("\n================ ENERGY & EMISSIONS (LoRA Inference) =================")
    print(f"Total CO2 emissions (kg): {emissions:.6f}")
    # ================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)
