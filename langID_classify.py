#!/usr/bin/env python3
"""
2_classify.py
Step 2: Read TXT files, Batch, Classify on GPU.
"""
import pandas as pd
import torch
import fasttext
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import csv
import sys
from tqdm import tqdm
from text_util import *  # Import updated utils

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    print(f"Loading models on {DEVICE}...")
    ft = fasttext.load_model("lid.176.bin")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)
    model.eval()  # Set to eval mode

    spellers = load_spellers()
    return ft, model, tokenizer, spellers


def main():
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config_langID.txt')

    INPUT_CSV = config.get('CLASSIFY', 'INPUT_CSV')
    TEXT_DIR = config.get('CLASSIFY', 'TEXT_DIR')
    OUTPUT_LINES_LOG = config.get('CLASSIFY', 'OUTPUT_LINES_LOG')
    BATCH_SIZE = config.getint('CLASSIFY', 'BATCH_SIZE')

    # 1. Setup
    Path(OUTPUT_LINES_LOG).parent.mkdir(parents=True, exist_ok=True)
    ft_model, ppl_model, ppl_tok, spellers = load_models()

    # 2. Check work already done (Resume capability)
    processed_keys = set()
    if Path(OUTPUT_LINES_LOG).exists():
        try:
            # Read just the file/page columns to skip done work
            existing = pd.read_csv(OUTPUT_LINES_LOG, usecols=['file', 'page'])
            processed_keys = set(zip(existing['file'].astype(str), existing['page'].astype(str)))
            print(f"Resuming: Found {len(processed_keys)} pages already processed.")
        except Exception:
            print("Could not read existing log, starting fresh or appending blindly.")

    # 3. Open Output Stream
    write_header = not Path(OUTPUT_LINES_LOG).exists()

    with open(OUTPUT_LINES_LOG, 'a', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(["file", "page", "line_num", "text", "lang", "score", "ppl", "cat"])

        # 4. Read Input List
        df = pd.read_csv(INPUT_CSV)

        # Batch Accumulators
        batch_lines = []
        batch_meta = []  # Stores (file_id, page_id, line_num, original_text)

        print("Starting classification loop...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            file_id = str(row['file'])
            page_id = str(row['page'])

            if (file_id, page_id) in processed_keys:
                continue

            txt_path = Path(TEXT_DIR) / file_id / f"{file_id}-{page_id}.txt"
            if not txt_path.exists():
                continue  # Skip missing files

            # Read text
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # Filter locally
                cat, clean = pre_filter_line(line)
                if cat != "Process":
                    # Write non-process lines immediately (no GPU needed)
                    writer.writerow([file_id, page_id, i, line.strip(), "N/A", 0, 0, cat, ""])
                    continue

                batch_lines.append(clean)
                batch_meta.append((file_id, page_id, i, line.strip()))

                # PROCESS BATCH
                if len(batch_lines) >= BATCH_SIZE:
                    run_batch(batch_lines, batch_meta, writer, ft_model, ppl_model, ppl_tok, spellers)
                    batch_lines = []
                    batch_meta = []

        # Final Batch
        if batch_lines:
            run_batch(batch_lines, batch_meta, writer, ft_model, ppl_model, ppl_tok, spellers)


def run_batch(lines, meta, writer, ft, ppl_model, tokenizer, spellers):
    # 1. PPL (Original)
    ppls = calculate_perplexity_batch(lines, ppl_model, tokenizer, DEVICE)

    # 2. FastText (Original)
    labels, scores = ft.predict(lines, k=1)
    langs = [l[0].replace("__label__", "") for l in labels]
    scores = [s[0] for s in scores]

    # 3. Correction & Re-Check (Logic simplified for brevity)
    # Ideally, you gather corrections and batch-run PPL again.
    # For speed, you might skip correction PPL if original is "Good Enough".

    # Here we do a simple sequential correct + heuristic for the example
    for i, txt in enumerate(lines):
        # ... logic to correct text ...
        # If strict batching is needed for correction PPL, collect them into a new list here
        # and run calculate_perplexity_batch again.

        # Placeholder for writing result
        writer.writerow([
            meta[i][0], meta[i][1], meta[i][2], meta[i][3],
            langs[i], f"{scores[i]:.4f}", f"{ppls[i]:.2f}",
            "Clear",  # Replace with actual categorize_line call
            ""
        ])


if __name__ == "__main__":
    main()