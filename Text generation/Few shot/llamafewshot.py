import sys
import re
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, confusion_matrix, classification_report)
from codecarbon import EmissionsTracker

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Define the few-shot examples with the desired raw output label: "not_sarcastic"
FEW_SHOT_EXAMPLES = [
    ("I just love it when the internet goes out right before my deadline.", "sarcastic"),
    ("The store is located right across the street.", "not_sarcastic"), 
    ("Oh sure, I don't mind starting my vacation with 12 hours of packing.", "sarcastic"),
    ("This coffee is exactly what I needed this morning.", "not_sarcastic"), 
    ("That's exactly what I meant to do.", "sarcastic"),
    ("Please remember to sign your name at the bottom of the form.", "not_sarcastic"), 
]

def map_label_from_text(gen: str) -> str:
    """Maps the raw generated text to a standardized classification label."""
    s = (gen or "").lower().strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"^[\"'`]+|[\"'`]+$", "", s)
    
    # *** FIX APPLIED HERE: Changed [_-\s] to [_\s-] to fix "bad character range" error ***
    if re.search(r'\bnot[_\s-]*sarcastic\b', s):
        return "not sarcastic"
    # ************************************************************************************
    
    if re.search(r'\bsarcasm\b', s) or re.search(r'\bsarcastic\b', s):
        return "sarcastic"
        
    if re.search(r'classification[:\s]*not[_\s-]*sarcastic', s):
        return "not sarcastic"
    if re.search(r'classification[:\s]*sarcastic', s):
        return "sarcastic"
        
    toks = s.split()
    if not toks:
        return "not sarcastic" 
        
    first_token = toks[0].rstrip(" .,:;!?").lower()

    # Check for single-token outputs, including the new combined token
    if first_token in ("not","no", "not_sarcastic", "not-sarcastic"):
        return "not sarcastic"
    if first_token in ("sarcastic","yes","y","yeah","yep","classification:","the","this"):
        return "sarcastic"
        
    return "not sarcastic" # Default fallback


def extract_generated_text(full_decoded: str, prompt: str, tokenizer=None, out_ids=None, inputs=None):
    prompt_text = (prompt or "").strip()
    gen_text = full_decoded
    
    # Note: The decoding logic here is slightly less robust than token-slicing but works
    # with the full decoded string as originally structured.
    if prompt_text and prompt_text in full_decoded:
        gen_text = full_decoded.split(prompt_text, 1)[1]

    gen_text = gen_text.strip()
    # Attempt to isolate the model's response content (after the last 'assistant' marker)
    gen_text = gen_text.rsplit('assistant', 1)[-1].strip()
    gen_text = " ".join(gen_text.split())
    gen_text = gen_text.rstrip(" .,:;!?\n\r\t")
    return gen_text

def resolve_output_path(output_arg: str) -> Path:
    out = Path(output_arg)
    if out.exists() and out.is_dir():
        out = out / "predictions.csv"
    else:
        if not out.suffix:
            out_dir = out
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / "predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def main():
    parser = argparse.ArgumentParser(description="Llama-3.2 few-shot CSV evaluator (batched)")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV (must contain id, text, label)")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save CSV with predictions (file or directory)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Hugging Face model id to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation (default: 4). Try 2 first on 6GB VRAM.")
    parser.add_argument("--max-input-length", type=int, default=512, help="Max input token length (default: 512). Lower to 256 if OOM.")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Tokens to generate per example (default: 20)")
    parser.add_argument("--save-generated", action="store_true", help="Save the raw generated text into the output CSV")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = resolve_output_path(args.output)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. No GPU detected. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print("Using device:", device)
    print("Model:", args.model)
    print("Input CSV:", csv_path)
    print("Output CSV:", out_path)
    print(f"Batch size: {args.batch_size}, max_input_length: {args.max_input_length}, max_new_tokens: {args.max_new_tokens}")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"   
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Note: dtype=torch.float16 and device_map='auto' are used for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float16, device_map='auto')
    model.eval()

    df = pd.read_csv(str(csv_path))
    cols_low = [c.lower() for c in df.columns]
    required = ("id", "text", "label")
    col_map = {}
    for want in required:
        if want in cols_low:
            col_map[want] = df.columns[cols_low.index(want)]
        else:
            print(f"ERROR: Input CSV must contain column '{want}' (case-insensitive). Found: {df.columns.tolist()}")
            sys.exit(1)

    preds = []
    golds = []
    gen_texts = [] if args.save_generated else None

    total = len(df)
    print("\nStarting batched few-shot inference...\n")

    rows = list(df.itertuples(index=False))
    batch_size = max(1, int(args.batch_size))

    # --- FEW-SHOT CONTEXT SETUP ---
    # System message requests the new "not_sarcastic" format
    system_message = {"role": "system", "content": "classify the user's text either 'sarcastic' or 'not_sarcastic' answer in one word. Use the following examples as a guide."}
    
    # Create the few-shot examples as an alternating list of user/assistant messages
    example_messages = []
    for example_text, example_label in FEW_SHOT_EXAMPLES:
        example_messages.append({"role": "user", "content": f"User text to classify: {example_text}"})
        example_messages.append({"role": "assistant", "content": example_label})
    # -----------------------------

    try:
        for start in tqdm(range(0, total, batch_size), desc="Batches"):
            batch_rows = rows[start:start + batch_size]

            prompts = []
            batch_golds = []
            for row in batch_rows:
                text = str(getattr(row, col_map["text"]))
                gold_raw = str(getattr(row, col_map["label"])).strip().lower()
                
                # Normalize the gold label to the standardized format 'sarcastic' / 'not sarcastic'
                gold = "sarcastic" if gold_raw in ("1", "sarcastic", "true", "yes", "y") else "not sarcastic"
                batch_golds.append(gold)
                
                # Construct the full messages list: System + Examples + Current Query
                messages = [system_message] + example_messages + [
                    {"role": "user", "content": f"User text to classify: {text}"}
                ]
                
                prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                prompts.append(prompt)

            inputs = tokenizer(prompts,return_tensors="pt",padding=True,truncation=True,max_length=int(args.max_input_length)).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    top_p=1.0,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs['input_ids'].shape[1]
            for i, out_ids in enumerate(out[:, input_len:]):
                gen_text_raw = tokenizer.decode(out_ids, skip_special_tokens=True)
                
                gen_text = " ".join(gen_text_raw.split()).strip().rstrip(" .,:;!?\n\r\t")
                pred = map_label_from_text(gen_text)
                
                preds.append(pred)
                golds.append(batch_golds[i])
                if args.save_generated:
                    gen_texts.append(gen_text)
                idx = start + i
                if idx<100:
                    tqdm.write(f"[{idx}] GOLD={batch_golds[i]} | PRED={pred} | GEN='{gen_text}'")

            torch.cuda.empty_cache()

    except RuntimeError as e:
        print("\nERROR during generation:", e)
        if "out of memory" in str(e).lower():
            print("\nCUDA out of memory. Suggestions:")
            print("- Reduce --batch-size (try 1 or 2).")
            print("- Reduce --max-input-length (try 256).")
        sys.exit(1)

    df_out = df.copy()
    df_out["predicted_label"] = preds
    if args.save_generated:
        df_out["generated_text"] = gen_texts
    df_out.to_csv(str(out_path), index=False)
    print(f"\nPredictions saved to: {out_path.resolve()}")

    print("\n==================== METRICS ====================")
    acc = accuracy_score(golds, preds)
    try:
        prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average='binary', pos_label='sarcastic', zero_division=0)
    except Exception:
        prec, rec, f1 = 0.0, 0.0, 0.0
    mcc = matthews_corrcoef(golds, preds) if len(set(golds)) > 1 else 0.0

    labels_order = ["sarcastic", "not sarcastic"]
    cm = confusion_matrix(golds, preds, labels=labels_order)
    report = classification_report(golds, preds, labels=labels_order, target_names=labels_order, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    print("\n==================== CONFUSION MATRIX ====================")
    print("Order of labels:", labels_order)
    print("Rows = Actual (Gold labels)")
    print("Cols = Predicted labels\n")
    print("            Predicted")
    print("            sarcastic     not sarcastic")
    print("Actual")
    print(f"sarcastic     {cm[0][0]:<12} {cm[0][1]}")
    print(f"not sarcastic {cm[1][0]:<12} {cm[1][1]}")

    print("\n==================== CLASSIFICATION REPORT ====================")
    print(report)

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "input_csv": str(csv_path),
        "output_csv": str(out_path.resolve()),
        "num_samples": int(total),
        "accuracy": float(acc),
        "precision_sarcastic": float(prec),
        "recall_sarcastic": float(rec),
        "f1_sarcastic": float(f1),
        "matthews_corrcoef": float(mcc),
        "confusion_matrix": {
            "labels_order": labels_order,
            "matrix": cm.tolist()
        },
        "label_support": {
            "sarcastic": int((pd.Series(golds) == "sarcastic").sum()),
            "not_sarcastic": int((pd.Series(golds) == "not sarcastic").sum())
        },
        "classification_report_text": report
    }

    metrics_path = out_path.parent / "metrics.json"
    try:
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"\nMetrics JSON saved to: {metrics_path.resolve()}")
    except Exception as e:
        print(f"\nWARNING: Failed to write metrics JSON: {e}")
    # ================= ENERGY & EMISSIONS RESULTS =================
    emissions = tracker.stop()

    print("\n==================== ENERGY & EMISSIONS ====================")
    print(f"Total CO2 emissions (kg): {emissions:.6f}")
# ==============================================================

if __name__ == "__main__":
    main()