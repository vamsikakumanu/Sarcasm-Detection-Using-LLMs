import argparse
import sys
import json
import re
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from codecarbon import EmissionsTracker

# ================= CONFIG =================
MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 5
BATCH_SIZE = 4
# ==========================================

def make_prompt(text: str) -> str:
    return (
        "Classify the following text as sarcastic or not_sarcastic with the help of below guided examples.\n\n"

        "Example 1:\n"
        "Text : I just love it when the internet goes out right before my deadline.\n"
        "Answer: sarcastic\n\n"

        "Example 2:\n"
        "Text : The store is located right across the street.\n"
        "Answer: not_sarcastic\n\n"

        "Example 3:\n"
        "Text : Oh sure, I don't mind starting my vacation with 12 hours of packing.\n"
        "Answer: sarcastic\n\n"

        "Example 4:\n"
        "Text : This coffee is exactly what I needed this morning.\n"
        "Answer: not_sarcastic\n\n"

        "Example 5:\n"
        "Text : That's exactly what I meant to do.\n"
        "Answer: sarcastic\n\n"

        "Example 6:\n"
        "Text : Please remember to sign your name at the bottom of the form.\n"
        "Answer: not_sarcastic\n\n"

        "Now, answer:\n"
        f"Text : \"{text}\"\n"
        "Answer in one word:"
    )


def normalize_prediction(text):
    s = text.lower().strip()
    s = re.sub(r'^(answer|label|the answer is|classification is)\s*:?\s*', '', s)

    if re.search(r'\bnot\s*[_-]?\s*sarcastic\b', s):
        return "not_sarcastic"
    if re.search(r'\bsarcastic\b', s):
        return "sarcastic"

    if s.startswith("yes"):
        return "sarcastic"
    if s.startswith("no"):
        return "not_sarcastic"

    return "not_sarcastic"


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
        description="OLMo-2-1B Zero-Shot Sarcasm Classification (Plain Prompt)"
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = resolve_output_path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model :", MODEL_NAME)
    print("Input :", csv_path)
    print("Output:", out_path)

    tracker = EmissionsTracker(
        project_name="OLMo2-ZeroShot-PlainPrompt",
        experiment_id="olmo2_plain",
        output_dir=str(out_path.parent),
        output_file="emissions.csv",
        log_level="error",
    )
    tracker.start()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    texts = df[cols["text"]].astype(str).tolist()
    labels_raw = df[cols["label"]].astype(str).tolist()
    total = len(df)

    preds, golds, clean_outs = [], [], []

    print(f"\nRunning inference | batch={BATCH_SIZE} | samples={total}\n")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Batches"):
        end = min(start + BATCH_SIZE, total)

        batch_texts = texts[start:end]
        batch_labels = labels_raw[start:end]

        batch_golds = [
            "sarcastic" if lbl.lower().strip() in ("1","sarcastic","true","yes","y")
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
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i, seq in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = seq[input_len:]

            answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            if len(answer) == 0:
                answer = "not_sarcastic"   # safe fallback
            else:
                answer = answer.splitlines()[0].strip()


            pred = normalize_prediction(answer)

            preds.append(pred)
            clean_outs.append(answer)

            idx = start + i
            if idx < 40:
                tqdm.write("-" * 60)
                tqdm.write(f"ROW {idx}")
                tqdm.write(f"TEXT : {batch_texts[i][:150]}")
                tqdm.write(f"GEN  : {answer}")
                tqdm.write(f"PRED : {pred} | GOLD: {batch_golds[i]}")

    df_out = df.copy()
    df_out["predicted_label"] = preds
    df_out["generated_text"] = clean_outs
    df_out.to_csv(out_path, index=False)

    acc = accuracy_score(golds, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        golds, preds, pos_label="sarcastic", average="binary", zero_division=0
    )
    mcc = matthews_corrcoef(golds, preds)
    cm = confusion_matrix(golds, preds, labels=["sarcastic","not_sarcastic"])
    report = classification_report(golds, preds, zero_division=0)

    print("\n========== METRICS ==========")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)
    print("MCC      :", mcc)
    print("\nConfusion Matrix:\n", cm)
    print("\nReport:\n", report)

    metrics_json = {
    "model": MODEL_NAME,
    "num_samples": total,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "mcc": mcc,
    "confusion_matrix": {
        "labels": ["sarcastic", "not_sarcastic"],
        "matrix": cm.tolist()
    },
    "classification_report": classification_report(
        golds, preds, output_dict=True, zero_division=0
    )
}

    json_path = out_path.with_suffix(".metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    print("\nMetrics JSON saved to:", json_path)


    emissions = tracker.stop()
    print("\nCO2 Emissions (kg):", emissions)

if __name__ == "__main__":
    main()
