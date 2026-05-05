import os
import json
import argparse

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    cohen_kappa_score
)

from codecarbon import EmissionsTracker

# ================= CONFIG =================
MODEL_NAME = "google/flan-t5-large"
MAX_INPUT_LENGTH = 512
MAX_LABEL_LENGTH = 32
BATCH_SIZE = 4
# ==========================================


# ---------------- PROMPT ----------------
def make_prompt(text):
    return (
        "Classify the following text as sarcastic or not sarcastic.\n"
        f'Text: "{text}"\n'
        "Answer: "
    )


# ---------------- LOGIT FUNCTION ----------------
def compute_logprob_batch_flan(prompts, label_text, tokenizer, model, device, max_input_len, max_label_len):

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len
    ).to(device)

    label_list = [label_text] * len(prompts)

    labels = tokenizer(
        text_target=label_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_label_len
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )

    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    scores = []

    for i in range(labels.size(0)):

        score = 0.0
        count = 0

        for t in range(labels.size(1)):

            token_id = labels[i, t].item()

            if token_id == tokenizer.pad_token_id:
                continue

            lp = log_probs[i, t, token_id].item()

            score += lp
            count += 1

        score = score / max(count, 1)
        scores.append(score)

    return scores


# ---------------- LOGIT INSPECTION ----------------
def inspect_sample(text, tokenizer, model, device, max_input_len=512, max_label_len=32):

    prompt = make_prompt(text)

    labels = [" sarcastic", " not sarcastic"]
    results = {}

    print("\n==============================")
    print("TEXT:", text)
    print("==============================")

    for label in labels:

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len
        ).to(device)

        label_ids = tokenizer(
            text_target=[label],
            return_tensors="pt",
            truncation=True,
            max_length=max_label_len
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=label_ids
            )

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        score = 0.0
        count = 0

        print(f"\n--- Label: {label.strip()} ---")

        for t in range(label_ids.size(1)):

            token_id = label_ids[0, t].item()

            if token_id == tokenizer.pad_token_id:
                continue

            token = tokenizer.decode([token_id])
            lp = log_probs[0, t, token_id].item()

            print(f"{token:15} | {lp:.4f}")

            score += lp
            count += 1

        score = score / max(count, 1)

        print(f"Final Score ({label.strip()}): {score:.4f}")
        results[label] = score

    margin = results[" sarcastic"] - results[" not sarcastic"]

    print("\n==============================")
    print(f"Margin: {margin:.4f}")
    print("==============================\n")

    return results


# ---------------- MAIN ----------------
def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="FLAN-T5-LoRA-Logit-Eval",
        output_dir=args.output_dir,
        output_file="emissions.csv",
        log_level="error"
    )
    tracker.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    base_model.config.decoder_start_token_id = tokenizer.pad_token_id

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    # 🔍 OPTIONAL DEBUG
    if args.inspect_text:
        inspect_sample(args.inspect_text, tokenizer, model, device)

    df = pd.read_csv(args.test_csv)

    preds, golds, margins, probs = [], [], [], []

    print("\n🚀 Starting evaluation...\n")

    for i in tqdm(range(0, len(df), BATCH_SIZE)):

        batch = df.iloc[i:i+BATCH_SIZE]

        texts = batch["text"].tolist()
        labels = batch["label"].tolist()

        prompts = [make_prompt(t) for t in texts]

        gold_batch = []
        for l in labels:
            l = str(l).lower().strip()
            gold_batch.append("sarcastic" if l in ["1","sarcastic","true","yes"] else "not sarcastic")

        golds.extend(gold_batch)

        s_scores = compute_logprob_batch_flan(
            prompts, " sarcastic", tokenizer, model, device, MAX_INPUT_LENGTH, MAX_LABEL_LENGTH
        )

        n_scores = compute_logprob_batch_flan(
            prompts, " not sarcastic", tokenizer, model, device, MAX_INPUT_LENGTH, MAX_LABEL_LENGTH
        )

        for s, n in zip(s_scores, n_scores):
            margin = s - n
            margins.append(margin)

            prob = 1 / (1 + np.exp(-margin))
            probs.append(prob)

    margins = np.array(margins)

    threshold = 0.0
    print(f"\nThreshold: {threshold:.4f}")

    for m in margins:
        preds.append("sarcastic" if m > threshold else "not sarcastic")

    # ---------------- METRICS ----------------
    acc = accuracy_score(golds, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        golds, preds, average="binary", pos_label="sarcastic", zero_division=0
    )

    mcc = matthews_corrcoef(golds, preds)
    kappa = cohen_kappa_score(golds, preds)

    gold_bin = [1 if g == "sarcastic" else 0 for g in golds]

    try:
        roc_auc = roc_auc_score(gold_bin, probs)
        fpr, tpr, _ = roc_curve(gold_bin, probs)
    except:
        roc_auc = 0.0
        fpr, tpr = [0, 1], [0, 1]

    cm = confusion_matrix(golds, preds)

    print("\n=========== RESULTS ===========")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Kappa     : {kappa:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:\n")
    print(classification_report(golds, preds, digits=4))

    # ---------------- SAVE ----------------
    df["prediction"] = preds
    df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "kappa": float(kappa),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- ROC ----------------
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "roc_auc.png"))
    plt.close()

    emissions = tracker.stop()
    print(f"\n🌱 CO2 emissions: {emissions:.6f} kg")


# ---------------- ENTRY ----------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--inspect_text", default=None)

    args = parser.parse_args()

    main(args)