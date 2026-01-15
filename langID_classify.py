#!/usr/bin/env python3
"""
2_classify.py
Step 2: Read TXT files, Batch, Classify on GPU.
Output: Individual CSV files per document.
"""
import pandas as pd
import torch
import fasttext
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import csv
import sys
from tqdm import tqdm
from itertools import groupby
import configparser
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


def write_rows_to_doc(output_dir, file_id, rows):
    """
    Appends rows to the specific document CSV.
    Creates the file and writes header if it doesn't exist.
    """
    out_path = Path(output_dir) / f"{file_id}.csv"
    file_exists = out_path.exists()

    with open(out_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # User specified header
            writer.writerow(["file", "page_num", "line_num", "text", "lang", "lang_score", "perplex", "categ"])
        writer.writerows(rows)


def main():
    # Initialize the parser
    config = configparser.ConfigParser()
    config.read('config_langID.txt')

    INPUT_CSV = config.get('CLASSIFY', 'INPUT_CSV')
    TEXT_DIR = config.get('CLASSIFY', 'TEXT_DIR')
    OUTPUT_DIR = config.get('CLASSIFY', 'OUTPUT_LINES_LOG')
    BATCH_SIZE = config.getint('CLASSIFY', 'BATCH_SIZE')

    # 1. Setup
    out_dir_path = Path(OUTPUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ft_model, ppl_model, ppl_tok, spellers = load_models()

    # 2. Read Input List
    df = pd.read_csv(INPUT_CSV)

    # Batch Accumulators
    batch_lines = []
    batch_meta = []  # Stores (file_id, page_id, line_num, original_text)

    # State for Rerun/Skip Logic
    current_file_id = None
    skipping_current_file = False

    # Track files we have started working on in *this* execution session
    # This ensures we don't skip files we just created a few seconds ago if the input dataframe is unsorted.
    session_files = set()

    print(f"Starting classification loop. Outputting to {OUTPUT_DIR}/...")
    print("Files with existing CSV outputs will be skipped.")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = str(row['file'])
        page_id = str(row['page'])

        # 3. Check Skip Logic (Per File)
        if file_id != current_file_id:
            current_file_id = file_id

            out_path = out_dir_path / f"{file_id}.csv"

            # If file exists on disk AND we haven't touched it in this session -> Skip it (Old run)
            if out_path.exists() and file_id not in session_files:
                skipping_current_file = True
            else:
                skipping_current_file = False
                session_files.add(file_id)

        if skipping_current_file:
            continue

        # 4. Process File
        txt_path = Path(TEXT_DIR) / file_id / f"{file_id}-{page_id}.txt"
        if not txt_path.exists():
            continue

        # Read text
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            clean_line = line.strip()
            # Filter locally
            cat, clean = pre_filter_line(line)

            if cat != "Process":
                # Write non-process lines immediately (no GPU needed)
                # Row format: "file", "page_num", "line_num", "text", "lang", "lang_score", "perplex", "categ"
                row_data = [file_id, page_id, i, clean_line, "N/A", 0, 0, cat]
                write_rows_to_doc(out_dir_path, file_id, [row_data])
                continue

            batch_lines.append(clean)
            batch_meta.append((file_id, page_id, i, clean_line))

            # PROCESS BATCH
            if len(batch_lines) >= BATCH_SIZE:
                process_and_write_batch(batch_lines, batch_meta, out_dir_path, ft_model, ppl_model, ppl_tok)
                batch_lines = []
                batch_meta = []

    # Final Batch
    if batch_lines:
        process_and_write_batch(batch_lines, batch_meta, out_dir_path, ft_model, ppl_model, ppl_tok)


def process_and_write_batch(lines, meta, out_dir, ft, ppl_model, tokenizer):
    """
    Runs models on the batch, matches results to metadata,
    groups by file_id, and writes to corresponding CSVs.
    """
    # 1. PPL
    ppls = calculate_perplexity_batch(lines, ppl_model, tokenizer, DEVICE)

    # 2. FastText
    labels, scores = ft.predict(lines, k=1)
    langs = [l[0].replace("__label__", "") for l in labels]
    scores = [s[0] for s in scores]

    # 3. Aggregate Results
    results = []
    for i in range(len(lines)):
        file_id, page_id, line_num, original_text = meta[i]

        # Format: "file", "page_num", "line_num", "text", "lang", "lang_score", "perplex", "categ"
        row = [
            file_id,
            page_id,
            line_num,
            original_text,
            langs[i],
            f"{scores[i]:.4f}",
            f"{ppls[i]:.2f}",
            categorize_line(langs[i], scores[i], ppls[i])
        ]
        results.append(row)

    # 4. Group by file_id and write
    # Sort by file_id first required for groupby
    results.sort(key=lambda x: x[0])

    for file_id, group in groupby(results, key=lambda x: x[0]):
        rows_for_file = list(group)
        write_rows_to_doc(out_dir, file_id, rows_for_file)


if __name__ == "__main__":
    main()