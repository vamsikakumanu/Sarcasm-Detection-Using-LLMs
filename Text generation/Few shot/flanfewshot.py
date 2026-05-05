import argparse
import sys
import re
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
    classification_report)
from codecarbon import EmissionsTracker


MODEL_NAME = "google/flan-t5-large"
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 4
BATCH_SIZE = 4 

def make_prompt(text: str) -> str:

    examples = [
        ("I just love it when the internet goes out right before my deadline.", "sarcastic"),
        ("The store is located right across the street.", "not sarcastic"),
        ("Oh sure, I don't mind starting my vacation with 12 hours of packing.", "sarcastic"),
        ("This coffee is exactly what I needed this morning.", "not sarcastic"),
        ("That's exactly what I meant to do.", "sarcastic"),
        ("Please remember to sign your name at the bottom of the form.", "not sarcastic"),
    ]

    example_string = ""
    for ex_text, ex_label in examples:
        example_string += f"Text: \"{ex_text}\"\n"
        example_string += f"Answer: {ex_label}\n\n"
    
    return (
        "Classify the following text as sarcastic or not_sarcastic. with the help of below guided examples \n\n"
        f"{example_string}"
        f"Text: \"{text}\"\n"
        "Answer:"
    )

def map_label(gen: str) -> str:
    s = (gen or "").lower().strip()
    
    if re.search(r'\bnot\s*sarcastic\b', s) or re.search(r'\bnot[-_ ]sarcastic\b', s) or s in ["not", "no"]:
        return "not_sarcastic"
    
    if re.search(r'\bsarcastic\b', s) or s in ["sarcastic", "yes", "sarcasm"]:
        return "sarcastic"
    return "not_sarcastic"

def resolve_output_path(output_arg: str) -> Path:
    """Resolves output path, creating directories if necessary."""
    p = Path(output_arg).resolve()
    if p.is_dir():
        p = p / "predictions.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def main():
    parser = argparse.ArgumentParser(description="Run Flan-T5 Large few-shot sarcasm classification.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to the input CSV file (must contain 'text' and 'label' columns).")
    parser.add_argument("--output", required=True, type=Path, help="Path to the output directory or CSV file for results.")
    args = parser.parse_args()

    csv_path = args.csv
    out_path = resolve_output_path(args.output)
    metrics_path = out_path.parent / "metrics.json"

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model and Tokenizer
    print(f"Loading model: {MODEL_NAME}")
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

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Load model in half-precision (float16) on GPU to save memory
        if device.type == 'cuda':
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 3. Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

    # Validate columns (case-insensitive)
    cols = [c.lower() for c in df.columns]
    if 'text' not in cols or 'label' not in cols:
        print("Error: Input CSV must contain columns 'text' and 'label'.")
        sys.exit(1)
    
    text_col = df.columns[cols.index('text')]
    label_col = df.columns[cols.index('label')]
    
    texts = df[text_col].astype(str).tolist()
    total = len(texts)
    
    print(f"Loaded {total} samples from {csv_path}. Starting inference...")

    # 4. Batch Inference
    raw_outputs = []
    clean_outputs = []

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Generating Predictions"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        # Create few-shot prompts for the batch
        batch_prompts = [make_prompt(text) for text in batch_texts]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            max_length=MAX_INPUT_LEN, 
            truncation=True, 
            padding=True
        ).to(device)

        with torch.no_grad():
            # Generate output (using greedy search for classification)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1
            )
        
        # Decode and process outputs
        decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        raw_outputs.extend(decoded_outputs)
        
        # Map raw output to standardized label
        clean_outputs.extend([map_label(gen) for gen in decoded_outputs])

    df['raw_model_output'] = raw_outputs
    df['predicted_label'] = clean_outputs

    # 5. Normalize Gold Labels for Evaluation
    golds = df[label_col].astype(str).str.lower().str.strip()
    def normalize_label(label: str) -> str:
        s = label.replace('_', ' ').replace('-', ' ')
        if s in ["sarcastic", "sarcasm", "1", "y", "yes", "true", "t"]:
            return "sarcastic"
        return "not_sarcastic"
        
    golds = golds.apply(normalize_label).tolist()
    
    # 6. Save Results
    df['gold_label'] = golds
    # ================== SAMPLE PREDICTIONS PREVIEW ==================
    PREVIEW_N = 20  # number of samples to preview

    print("\n==================== PREDICTIONS PREVIEW ====================")
    print(f"Total samples: {total}")

    for i in range(min(PREVIEW_N, total)):
        print("\n------------------------------------------------------------")
        print(f"SAMPLE INDEX   : {i}")
        print(f"TEXT           : {df.iloc[i][text_col]}")
        print(f"MODEL OUTPUT   : {df.iloc[i]['raw_model_output']}")
        print(f"PREDICTED LABEL: {df.iloc[i]['predicted_label']}")
        print(f"GOLD LABEL     : {df.iloc[i]['gold_label']}")

    print("\n==================== END PREVIEW ====================")
# ====================================================


    preds = df['predicted_label'].tolist()
    
    labels_order = ["sarcastic", "not_sarcastic"]
    
    acc = accuracy_score(golds, preds)
    
    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
        golds, 
        preds, 
        labels=labels_order, 
        average=None, 
        zero_division=0.0
    )
    
    # Map the per-class results for easier access
    metrics_map = {
        label: {"precision": p, "recall": r, "f1": f, "support": s}
        for label, p, r, f, s in zip(labels_order, precisions, recalls, f1_scores, supports)
    }

    mcc = matthews_corrcoef(golds, preds)
    cm = confusion_matrix(golds, preds, labels=labels_order)
    report = classification_report(golds, preds, labels=labels_order, zero_division=0.0)

    print("\n======================= CLASSIFICATION METRICS ======================")
    print(f"Total Samples: {total}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    
    # NEW: Display detailed per-class metrics
    print("\n--- Sarcastic (Positive Class) ---")
    print(f"Precision: {metrics_map['sarcastic']['precision']:.4f}")
    print(f"Recall:    {metrics_map['sarcastic']['recall']:.4f}")
    print(f"F1-Score:  {metrics_map['sarcastic']['f1']:.4f}")
    
    print("\n--- Not Sarcastic (Negative Class) ---")
    print(f"Precision: {metrics_map['not_sarcastic']['precision']:.4f}")
    print(f"Recall:    {metrics_map['not_sarcastic']['recall']:.4f}")
    print(f"F1-Score:  {metrics_map['not_sarcastic']['f1']:.4f}")
    
    print("\n====================== CONFUSION MATRIX =======================")
    header = f"{labels_order[0]:<12} {labels_order[1]}"
    print("            Predicted")
    print(header)
    print("Actual")
    print(f"{labels_order[0]:<15} {cm[0][0]:<12} {cm[0][1]}")
    print(f"{labels_order[1]:<15} {cm[1][0]:<12} {cm[1][1]}")

    print("\n==================== CLASSIFICATION REPORT ====================")
    print(report)

    # 8. Save metrics to JSON (Updated structure for precision/recall)
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": MODEL_NAME,
        "input_csv": str(csv_path),
        "output_csv": str(out_path.resolve()),
        "num_samples": int(total),
        "accuracy": float(acc),
        "matthews_corrcoef": float(mcc),
        "per_class_metrics": {
            "sarcastic": {
                "precision": float(metrics_map['sarcastic']['precision']),
                "recall": float(metrics_map['sarcastic']['recall']),
                "f1": float(metrics_map['sarcastic']['f1']),
                "support": int(metrics_map['sarcastic']['support']),
            },
            "not_sarcastic": {
                "precision": float(metrics_map['not_sarcastic']['precision']),
                "recall": float(metrics_map['not_sarcastic']['recall']),
                "f1": float(metrics_map['not_sarcastic']['f1']),
                "support": int(metrics_map['not_sarcastic']['support']),
            },
        },
        "confusion_matrix": {
            "labels_order": labels_order,
            "matrix": cm.tolist()
        },
        "classification_report_text": report
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nMetrics saved to {metrics_path.resolve()}")
    # ================= ENERGY & EMISSIONS RESULTS =================
    emissions = tracker.stop()

    print("\n==================== ENERGY & EMISSIONS ====================")
    print(f"Total CO2 emissions (kg): {emissions:.6f}")
# ==============================================================

if __name__ == "__main__":
    main()