#!/usr/bin/env python3
"""
1_extract.py
Step 1: Extract text from ALTO XML files in parallel.
"""
import pandas as pd
import subprocess
import concurrent.futures
import os
import sys
from pathlib import Path
from tqdm import tqdm
import configparser


def extract_single_page(args):
    """Worker function to extract one page."""
    file_id, page_id, xml_path, output_dir = args

    # Define output path
    # Using a flat directory structure or hashed structure is better for millions of files,
    # but adhering to original logic:
    save_dir = Path(output_dir) / str(file_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"{file_id}-{page_id}.txt"

    # Skip if exists
    if txt_path.exists():
        return True

    # Run extraction (alto-tools)
    cmd = ["alto-tools", "-t", xml_path]
    backup_xml_path = Path(xml_path).parents[1] / "onepagers" / Path(xml_path).name
    if backup_xml_path.exists():
        cmd = ["alto-tools", "-t", str(backup_xml_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if res.returncode == 0:
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            return True
        else:
            return False
    except Exception:
        return False


def main():
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config_langID.txt')

    INPUT_CSV = config.get('EXTRACT', 'INPUT_CSV')
    OUTPUT_TEXT_DIR = config.get('EXTRACT', 'OUTPUT_TEXT_DIR')
    MAX_WORKERS = config.getint('EXTRACT', 'MAX_WORKERS')

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} pages to extract.")

    tasks = []
    for _, row in df.iterrows():
        tasks.append((row['file'], row['page'], row['path'], OUTPUT_TEXT_DIR))

    # Parallel Execution
    print(f"Extracting with {MAX_WORKERS} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(extract_single_page, tasks), total=len(tasks)))

    print(f"Extraction complete. Success rate: {sum(results) / len(results):.2%}")


if __name__ == "__main__":
    main()