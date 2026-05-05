
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from codecarbon import EmissionsTracker
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

DEFAULT_MODEL = "google/flan-t5-large"
BATCH_SIZE = 2


# ---------------- PROMPT ----------------
def make_prompt(text: str):
    return (
        "Classify the following text as sarcastic or not sarcastic.\n"
        f"Text: {text}\n"
        "Answer:"
    )


# ---------------- OUTPUT PATH ----------------
def resolve_output_path(output_arg: str):
    out = Path(output_arg)

    if out.exists() and out.is_dir():
        out = out / "predictions.csv"
    elif not out.suffix:
        out.mkdir(parents=True, exist_ok=True)
        out = out / "predictions.csv"

    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# ---------------- LOGPROB (UNCHANGED) ----------------
def compute_logprob_batch(prompts, label_text, tokenizer, model, device, max_len):

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    ).to(device)

    labels = tokenizer(
        [label_text] * len(prompts),
        return_tensors="pt",
        padding=True
    ).input_ids.to(device)

    labels[labels == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    scores = []

    for i in range(labels.size(0)):
        score = 0.0
        count = 0

        for t in range(labels.size(1)):
            token_id = labels[i, t].item()

            if token_id == -100:
                continue

            score += log_probs[i, t, token_id].item()
            count += 1

        score = score / max(count, 1)
        scores.append(score)

    return scores


# ---------------- MAIN ----------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-input-length", type=int, default=512)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.")
        sys.exit(1)

    device = torch.device("cuda")

    print("\nLoading model:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16
    ).to(device).eval()

    df = pd.read_csv(args.csv)

    rows = list(df.itertuples(index=False))
    out_path = resolve_output_path(args.output)

    preds = []
    golds = []
    prob_sarcastic = []
    margins = []

    # ---------------- CODECARBON ----------------
    tracker = EmissionsTracker(
        project_name="flant5-logit",
        output_dir=str(out_path.parent),
        output_file="emissions.csv",
        log_level="error"
    )

    tracker.start()

    print("\nStarting FIRST PASS (logit scoring)...\n")

    all_s_scores = []
    all_n_scores = []

    # -------- FIRST PASS --------
    for i in tqdm(range(0, len(rows), BATCH_SIZE)):

        batch = rows[i:i + BATCH_SIZE]

        texts = [str(r.text) for r in batch]
        prompts = [make_prompt(t) for t in texts]

        gold_batch = []

        for r in batch:
            gold_raw = str(r.label).lower().strip()
            gold = (
                "sarcastic"
                if gold_raw in ("1", "sarcastic", "true", "yes", "y")
                else "not sarcastic"
            )
            gold_batch.append(gold)

        golds.extend(gold_batch)

        s_scores = compute_logprob_batch(
            prompts, "sarcastic", tokenizer, model, device, args.max_input_length
        )

        n_scores = compute_logprob_batch(
            prompts, "not sarcastic", tokenizer, model, device, args.max_input_length
        )

        all_s_scores.extend(s_scores)
        all_n_scores.extend(n_scores)

    # -------- CALIBRATION --------
    all_s_scores = np.array(all_s_scores)
    all_n_scores = np.array(all_n_scores)

    margins = all_s_scores - all_n_scores
    threshold = np.median(margins)

    print(f"\nCalibrated threshold: {threshold:.4f}")

    # -------- SECOND PASS --------
    print("\nStarting SECOND PASS (prediction)...\n")

    for i, (s, n, gold) in enumerate(zip(all_s_scores, all_n_scores, golds)):

        margin = s - n

        pred = "sarcastic" if margin > threshold else "not sarcastic"
        preds.append(pred)

        prob = 1 / (1 + np.exp(-margin))
        prob_sarcastic.append(prob)

        if i < 50:
            print(f"[{i}] S={s:.3f} | N={n:.3f} | M={margin:.3f} | PRED={pred} | GOLD={gold}")

    # -------- SAVE --------
    df_out = df.copy()
    df_out["predicted_label"] = preds
    df_out.to_csv(out_path, index=False)

    print("\nPredictions saved to:", out_path.resolve())

    # -------- METRICS --------
    acc = accuracy_score(golds, preds)

    prec, rec, f1, _ = precision_recall_fscore_support(
        golds, preds,
        average="binary",
        pos_label="sarcastic",
        zero_division=0
    )

    mcc = matthews_corrcoef(golds, preds)

    cm = confusion_matrix(golds, preds, labels=["sarcastic", "not sarcastic"])

    report = classification_report(
        golds, preds,
        labels=["sarcastic", "not sarcastic"],
        target_names=["sarcastic", "not sarcastic"],
        zero_division=0
    )

    golds_binary = [1 if g == "sarcastic" else 0 for g in golds]
    roc_auc = roc_auc_score(golds_binary, prob_sarcastic)
    kappa = cohen_kappa_score(golds_binary, [1 if p == "sarcastic" else 0 for p in preds])

    print("\n=========== METRICS ===========")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Kappa     : {kappa:.4f}")

    print("\n=========== CONFUSION MATRIX ===========")
    print(cm)

    print("\n=========== CLASSIFICATION REPORT ===========")
    print(report)

    # -------- SAVE ROC CURVE --------
    fpr, tpr, _ = roc_curve(golds_binary, prob_sarcastic)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = out_path.parent / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    print(f"\nROC curve saved to: {roc_path}")

    # -------- SAVE METRICS JSON --------
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "num_samples": len(rows),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "kappa": float(kappa),
        "threshold": float(threshold)
    }

    metrics_path = out_path.parent / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")

    # ---------------- STOP TRACKER ----------------
    tracker.stop()


if __name__ == "__main__":
    main()

