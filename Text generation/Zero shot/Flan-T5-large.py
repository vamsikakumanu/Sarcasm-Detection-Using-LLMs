import argparse
import sys
import json
from pathlib import Path
from datetime import datetime           

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from codecarbon import EmissionsTracker

# ================= CONFIG =================
MODEL_NAME = "google/flan-t5-large"
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 4
BATCH_SIZE = 4
# ==========================================


def make_prompt(text: str) -> str:
    return (
        "Classify the following text as sarcastic or not_sarcastic.\n"
        f'Text: "{text}"\n'
        "Answer :"
    )


def normalize_prediction(text):
    text = text.strip().lower()
    if text.startswith("not_sarcastic") or text.startswith("not sarcastic") or text.startswith("not"):
        return "not_sarcastic"
    elif text.startswith("sarcastic") or text.startswith("sarc"):
        return "sarcastic"
    else:
        return "unknown"
# ==================================================


def resolve_output_path(output_arg: str) -> Path:
    out = Path(output_arg)
    if out.exists() and out.is_dir():
        out = out / "predictions.csv"
    else:
        if not out.suffix:
            out.mkdir(parents=True, exist_ok=True)
            out = out / "predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Flan-T5 batch inference with user-defined normalization"
    )
    parser.add_argument("--csv", required=True, help="Input CSV with text & label")
    parser.add_argument("--output", required=True, help="Output file or directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("ERROR: CSV not found:", csv_path)
        sys.exit(1)

    out_path = resolve_output_path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model :", MODEL_NAME)
    print("Input :", csv_path)
    print("Output:", out_path)

    # -------- Load model --------
    # ================= ENERGY & EMISSIONS TRACKING =================
    tracker = EmissionsTracker(
    project_name="Qwen2.5-ZeroShot-Sarcasm",
    experiment_id="qwen_zeroshot_eval",
    output_dir=str(out_path.parent),
    output_file="emissions.csv",
    measure_power_secs=1,
    log_level="error"
)

    tracker.start()
# ===============================================================

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    # -------- Load data --------
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    if "text" not in cols or "label" not in cols:
        print("ERROR: CSV must contain 'text' and 'label' columns")
        sys.exit(1)

    texts = df[cols["text"]].astype(str).tolist()
    labels_raw = df[cols["label"]].astype(str).tolist()
    total = len(df)

    preds, raw_outs, clean_outs, golds = [], [], [], []

    print(f"\nRunning inference | batch={BATCH_SIZE} | samples={total}\n")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Batches"):
        end = min(start + BATCH_SIZE, total)

        batch_texts = texts[start:end]
        batch_labels = labels_raw[start:end]

        # normalize gold labels
        batch_golds = [
            "sarcastic"
            if lbl.strip().lower() in ("1", "sarcastic", "true", "yes", "y")
            else "not_sarcastic"
            for lbl in batch_labels
        ]
        golds.extend(batch_golds)

        prompts = [make_prompt(t) for t in batch_texts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i, seq in enumerate(outputs):
            raw = tokenizer.decode(seq, skip_special_tokens=False)
            clean = tokenizer.decode(seq, skip_special_tokens=True)

            pred = normalize_prediction(clean)

            preds.append(pred)
            raw_outs.append(raw)
            clean_outs.append(clean)

            idx = start + i
            if idx < 50:
                tqdm.write("-" * 60)
                tqdm.write(f"ROW {idx}")
                tqdm.write(f"TEXT : {batch_texts[i][:200]}")
                tqdm.write(f"CLEAN: {repr(clean)}")
                tqdm.write(f"PRED : {pred} | GOLD: {batch_golds[i]}")

    # -------- Save predictions --------
    df_out = df.copy()
    df_out["predicted_label"] = preds
    df_out["clean_model_output"] = clean_outs
    df_out["raw_model_output"] = raw_outs
    df_out.to_csv(out_path, index=False)

    print("\nPredictions saved to:", out_path.resolve())

    # ===== METRICS =====
    # Replace "unknown" safely for metrics
    preds_for_metrics = [
        p if p in ("sarcastic", "not_sarcastic") else "not_sarcastic"
        for p in preds
    ]

    labels_order = ["sarcastic", "not_sarcastic"]

    acc = accuracy_score(golds, preds_for_metrics)
    prec, rec, f1, _ = precision_recall_fscore_support(
        golds,
        preds_for_metrics,
        pos_label="sarcastic",
        average="binary",
        zero_division=0,
    )
    mcc = matthews_corrcoef(golds, preds_for_metrics)
    cm = confusion_matrix(golds, preds_for_metrics, labels=labels_order)
    report = classification_report(
        golds,
        preds_for_metrics,
        labels=labels_order,
        target_names=labels_order,
        zero_division=0,
    )

    print("\n========== METRICS ==========")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": MODEL_NAME,
        "samples": total,
        "accuracy": acc,
        "precision_sarcastic": prec,
        "recall_sarcastic": rec,
        "f1_sarcastic": f1,
        "mcc": mcc,
        "confusion_matrix": {
            "labels": labels_order,
            "matrix": cm.tolist(),
        },
        "classification_report": report,
        "unknown_predictions": int(sum(p == "unknown" for p in preds)),
    }

    metrics_path = out_path.parent / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics saved to:", metrics_path.resolve())
    # ================= ENERGY & EMISSIONS RESULTS =================
    emissions = tracker.stop()

    print("\n==================== ENERGY & EMISSIONS ====================")
    print(f"Total CO2 emissions (kg): {emissions:.6f}")
# ==============================================================


if __name__ == "__main__":
    main()
