import sys
import re
import argparse
import json
from pathlib import Path
from datetime import datetime
from codecarbon import EmissionsTracker

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

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ======================= FEW-SHOT PROMPT =======================
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


def map_label_from_text(gen: str) -> str:
    s = (gen or "").lower().strip()

    # remove common prefixes
    s = re.sub(
        r'^(human|assistant|answer|the answer is|model|classification)\s*:?\s*',
        '',
        s
    )

    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())

    # priority: not sarcastic
    if re.search(r'\bnot\s+sar', s):
        return "not sarcastic"

    if re.search(r'\bsarcastic\b', s):
        return "sarcastic"
    if re.search(r'\bsarcasm\b', s):
        return "sarcastic"

    if s.startswith("not"):
        return "not sarcastic"

    return "not sarcastic"


def extract_generated_text(full_decoded: str, prompt: str) -> str:
    if prompt in full_decoded:
        gen = full_decoded.split(prompt, 1)[1]
    else:
        gen = full_decoded
    gen = gen.strip()
    gen = " ".join(gen.split())
    gen = gen.rstrip(" .,:;!?\n\r\t")
    return gen


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
    parser = argparse.ArgumentParser(description="Qwen Few-Shot CSV Evaluator")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--save-generated", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("CUDA GPU required.")
        sys.exit(1)
    out_path = resolve_output_path(args.output)
    

    tracker = EmissionsTracker(
    project_name="Qwen2.5-fewShot-Sarcasm",
    experiment_id="qwen_zeroshot_eval",
    output_dir=str(out_path.parent),
    output_file="emissions.csv",
    measure_power_secs=1,
    log_level="error")

    tracker.start()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float16
    ).to(device).eval()

    df = pd.read_csv(args.csv)

    preds, golds, gens = [], [], []

    rows = list(df.itertuples(index=False))

    for start in tqdm(range(0, len(rows), args.batch_size), desc="Batches"):
        batch = rows[start:start + args.batch_size]

        prompts = []
        batch_golds = []

        for r in batch:
            text = str(r.text)
            gold_raw = str(r.label).lower()
            gold = "sarcastic" if gold_raw in ("1", "sarcastic") else "not sarcastic"
            batch_golds.append(gold)
            prompts.append(make_prompt(text))

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i, out_ids in enumerate(outputs):
            full = tokenizer.decode(out_ids, skip_special_tokens=True)
            gen_text = extract_generated_text(full, prompts[i])
            pred = map_label_from_text(gen_text)

            preds.append(pred)
            golds.append(batch_golds[i])
            gens.append(gen_text)

            idx = start + i
            if idx < 50:
                tqdm.write(f"[{idx}] GOLD={batch_golds[i]} | PRED={pred} | GEN='{gen_text}'")

        torch.cuda.empty_cache()

    out_path = resolve_output_path(args.output)

    df_out = df.copy()
    df_out["predicted_label"] = preds
    if args.save_generated:
        df_out["generated_text"] = gens
    df_out.to_csv(out_path, index=False)

    acc = accuracy_score(golds, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        golds, preds, average="binary", pos_label="sarcastic", zero_division=0
    )
    mcc = matthews_corrcoef(golds, preds)

    cm = confusion_matrix(golds, preds, labels=["sarcastic", "not sarcastic"])
    report = classification_report(
        golds, preds,
        labels=["sarcastic", "not sarcastic"],
        zero_division=0
    )

    print("\nAccuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("MCC:", mcc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }

    with open(out_path.parent / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    emissions = tracker.stop()

    print("\n==================== ENERGY & EMISSIONS ====================")
    print(f"Total CO2 emissions (kg): {emissions:.6f}")



if __name__ == "__main__":
    main()
