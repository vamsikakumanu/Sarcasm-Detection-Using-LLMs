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

MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

PROMPT = (
    "Classify the following text as sarcastic or not_sarcastic.\n"
    "Text: {text}\n"
    "Answer:"
)

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 10
BATCH_SIZE = 4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_prediction(text: str) -> str:
    if text is None:
        return "unknown"

    s = text.lower().strip()
    s = re.sub(r'(answer:|classification:)', '', s).strip()

    if s.startswith("not_sarcastic") or s.startswith("not sarcastic") or s.startswith("not") or s.startswith("ot_sarcastic"):
        return "not_sarcastic"
    if s.startswith("sarcastic") or s.startswith("sarc"):
        return "sarcastic"
    return "unknown"


def main(args):
    set_seed(SEED)

    tracker = EmissionsTracker(
        project_name="OLMo2-LoRA-Inference",
        experiment_id="lora_adapter_testing",
        output_dir=args.output_dir,
        output_file="lora_test_emissions.csv",
        log_level="error"
    )
    tracker.start()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(args.test_csv)
    print(f"Loaded test samples: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    gold_labels, pred_labels, raw_generations = [], [], []

    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[start:start + BATCH_SIZE]

        texts = batch["text"].tolist()
        golds = batch["label"].tolist()

        prompts = [PROMPT.format(text=t) for t in texts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        decoded_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, raw_answer in enumerate(decoded_batch):
            pred = normalize_prediction(raw_answer.strip())

            gold_labels.append(
                "sarcastic" if str(golds[i]).lower() in ["1","sarcastic","true","yes"] else "not_sarcastic"
            )
            pred_labels.append(pred)
            raw_generations.append(raw_answer.strip())

    # Filter unknowns
    filtered_gold, filtered_pred = [], []
    for g, p in zip(gold_labels, pred_labels):
        if p != "unknown":
            filtered_gold.append(g)
            filtered_pred.append(p)

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
    report = classification_report(filtered_gold, filtered_pred, zero_division=0)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save CSV
    df_out = df.copy()
    df_out["prediction"] = pred_labels
    df_out["raw_generation"] = raw_generations
    df_out.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # Save JSON metrics
    metrics = {
        "model": MODEL_NAME,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(filtered_gold, filtered_pred, output_dict=True),
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n================= METRICS =================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print("\nConfusion Matrix [sarcastic, not_sarcastic]:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    emissions = tracker.stop()
    print("\n================ ENERGY & EMISSIONS =================")
    print(f"Total CO2 emissions (kg): {emissions:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", type=str, default="test_split_40.csv")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)
