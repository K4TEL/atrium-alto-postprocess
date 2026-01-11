#!/usr/bin/env python3
"""
3_aggregate.py
Step 3: Aggregate raw lines into page statistics and split per document.
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import configparser

def main():
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the configuration file
    config.read('config_langID.txt')

    RAW_LINES_CSV = config.get('AGGREGATE', 'RAW_LINES_CSV')
    OUTPUT_STATS = config.get('AGGREGATE', 'OUTPUT_STATS')
    OUTPUT_DOC_DIR = config.get('AGGREGATE', 'OUTPUT_DOC_DIR')
    CHUNKSIZE = config.getint('AGGREGATE', 'CHUNKSIZE')

    Path(OUTPUT_DOC_DIR).mkdir(parents=True, exist_ok=True)

    # We need to aggregate counts per (file, page)
    # Since the input is sorted/grouped by file processing order, we can stream it.

    print(f"Reading {RAW_LINES_CSV} in chunks...")

    # Initialize global stats collector (be careful with memory here)
    # If dataset is truly massive, use Dask. For < 50M rows, chunking is usually fine.

    page_stats_list = []

    for chunk in tqdm(pd.read_csv(RAW_LINES_CSV, chunksize=CHUNKSIZE)):
        # 1. Pivot / Groupby on the chunk
        # Note: This assumes a page's lines don't span across chunks.
        # Since we write sequentially in Step 2, this is mostly safe,
        # but robust code handles edge cases.

        stats = chunk.groupby(['file', 'page'])['cat'].value_counts().unstack(fill_value=0)
        stats.reset_index(inplace=True)

        # Add missing columns if they didn't appear in this chunk
        for col in ["Clear", "Trash", "Noisy", "Empty", "Non-text"]:
            if col not in stats.columns:
                stats[col] = 0

        # Save page stats
        page_stats_list.append(stats)

        # 2. Split into per-document files immediately (optional)
        # Group by file and append to CSVs in OUTPUT_DOC_DIR
        for file_id, group in chunk.groupby('file'):
            doc_path = Path(OUTPUT_DOC_DIR) / f"lines_{file_id}.csv"
            header = not doc_path.exists()
            group.to_csv(doc_path, mode='a', index=False, header=header)

    # Combine all page stats
    print("Consolidating page stats...")
    final_df = pd.concat(page_stats_list)
    # Group again in case a page was split across chunks
    final_df = final_df.groupby(['file', 'page']).sum().reset_index()

    final_df.to_csv(OUTPUT_STATS, index=False)
    print(f"Done. Stats saved to {OUTPUT_STATS}")


if __name__ == "__main__":
    main()