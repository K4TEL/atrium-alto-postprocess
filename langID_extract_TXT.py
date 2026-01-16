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
    """Worker function to extract one page with robust de-hyphenation."""
    file_id, page_id, xml_path, output_dir = args

    # Define output path
    save_dir = Path(output_dir) / str(file_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"{file_id}-{page_id}.txt"

    # Skip if exists
    if txt_path.exists():
        return True

    # Define common hyphen variations found in OCR/Typesetting
    # Standard hyphen, Soft hyphen (\xad), En dash (\u2013), Em dash (\u2014)
    HYPHEN_VARIATIONS = ('-', '\xad', '\u2013', '\u2014')

    # Run extraction (alto-tools)
    cmd = ["alto-tools", "-t", xml_path]
    backup_xml_path = Path(xml_path).parents[1] / "onepagers" / Path(xml_path).name
    if backup_xml_path.exists():
        cmd = ["alto-tools", "-t", str(backup_xml_path)]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if res.returncode == 0:
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]

            # De-hyphenation Logic
            for i in range(len(lines) - 1):
                # Check if line ends with any of the hyphen variations
                if lines[i].endswith(HYPHEN_VARIATIONS):

                    # Remove the specific hyphen character detected
                    # We strip the last character regardless of which variation it was
                    prefix = lines[i][:-1]

                    next_line_parts = lines[i + 1].split(maxsplit=1)

                    if next_line_parts:
                        suffix = next_line_parts[0]

                        # Combine prefix and suffix on the current line
                        lines[i] = prefix + suffix

                        # Remove the suffix from the next line
                        if len(next_line_parts) > 1:
                            lines[i + 1] = next_line_parts[1]
                        else:
                            lines[i + 1] = ""

            # Final cleanup: Remove any empty lines created by the merge
            final_lines = [l for l in lines if l.strip()]

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(final_lines))
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