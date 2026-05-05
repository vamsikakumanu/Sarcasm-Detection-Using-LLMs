import os
import argparse
import json
import random
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from tqdm.auto import tqdm

# ---------------- FIXED (MUST MATCH TRAINING) ----------------
MODEL_NAME = "google/flan-t5-large"
PROMPT = (
    "classify the following text as sarcastic or not_sarcastic.\n"
    "Text: {text}\n"
    "Answer:"
)
MAX_INPUT_LENGTH = 256
SEED = 42
# ------------------------------------------------------------


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def normalize_prediction(text):
    text = text.strip().lower()
    if text.startswith("not_sarcastic") or text.startswith("not sarcastic") or text.startswith("not"):
        return "not_sarcastic"
    elif text.startswith("sarcastic") or text.startswith("sarc"):
        return "sarcastic"
    else:
        return "unknown"


def normalize_gold(label):
    label = label.strip().lower()
    return "not_sarcastic" if "not" in label else "sarcastic"


def evaluate_split(split_ds, model, tokenizer, batch_size, split_name, output_dir):
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in tqdm(range(0, len(split_ds), batch_size), desc=f"Evaluating {split_name}"):
        batch = split_ds[i:i + batch_size]

        inputs = [PROMPT.format(text=t) for t in batch["text"]]
        enc = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=6,
                do_sample=False,
                temperature=0.0,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text, gold, pred in zip(batch["text"], batch["label"], decoded):
            rows.append({
                "text": text,
                "gold_label": normalize_gold(gold),
                "predicted_label": normalize_prediction(pred),
                "raw_prediction": pred,
            })

    df = pd.DataFrame(rows)
    df_eval = df[df["predicted_label"] != "unknown"]

    y_true = df_eval["gold_label"]
    y_pred = df_eval["predicted_label"]

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label="sarcastic", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["sarcastic", "not_sarcastic"])
    report = classification_report(
        y_true,
        y_pred,
        labels=["sarcastic", "not_sarcastic"],
        digits=4,
        zero_division=0,
    )

    metrics = {
        "split": split_name,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(split_ds),
        "used_samples": len(df_eval),
        "ignored_unknowns": len(df) - len(df_eval),
    }

    # -------- Save outputs --------
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{split_name}_predictions.csv"), index=False)
    with open(os.path.join(output_dir, f"{split_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # -------- Console output --------
    print(f"\n===== {split_name.upper()} METRICS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print("\nConfusion Matrix [sarcastic, not_sarcastic]:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    set_seed(SEED)

    # -------- Load dataset --------
    ds = load_dataset("csv", data_files=args.data)["train"].shuffle(seed=SEED)
    n = len(ds)
    #train_ds = ds.select(range(int(0.6 * n)))   # SEEN
    test_ds = ds.select(range(int(0.6 * n), n)) # UNSEEN

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -------- Load model + adapter --------
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    #evaluate_split(train_ds, model, tokenizer, args.batch_size, "seen", args.output_dir)
    evaluate_split(test_ds, model, tokenizer, args.batch_size, "unseen", args.output_dir)


if __name__ == "__main__":
    main()
