import os
import csv
import argparse
import sys
import re
import multiprocessing
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try to import multi_rake; handle missing dependency gracefully
try:
    from multi_rake import Rake
except ImportError:
    print("[Error] 'multi_rake' library not found. Please install it: pip install multi_rake", file=sys.stderr)
    sys.exit(1)

# --- Configuration for "One-pagers" ---
# Regex to extract Document ID from filename stem.
# Pattern: (Group 1: DocName) [separator - or _] (Group 2: PageNum) at end of string
ONEPAGER_DOC_REGEX = re.compile(r"(.*)[-_](\d+)$")


def get_text_from_file(file_path: str) -> list[str]:
    """Reads a text file line by line."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []


def process_document_task(task_data):
    """
    Worker function to process a single document.
    Args: task_data (tuple): (doc_id, file_paths, lang, max_w_count, chunk_size)
    """
    doc_id, file_paths, lang, max_words, chunk_size = task_data

    try:
        # Initialize RAKE locally per process
        rake = Rake(language_code=lang, max_words=max_words)
    except Exception:
        return None

    aggregated_scores = defaultdict(float)
    current_lines = []

    # Sort files to maintain page order
    sorted_files = sorted(file_paths)

    for file_path in sorted_files:
        lines = get_text_from_file(str(file_path))
        if not lines:
            continue

        current_lines.extend(lines)

        # Process in chunks
        if len(current_lines) >= chunk_size:
            text_chunk = " ".join(current_lines)
            kw_scores = rake.apply(text_chunk)
            for kw, score in kw_scores:
                aggregated_scores[kw] += score
            current_lines = []

            # Process remaining lines
    if current_lines:
        text_chunk = " ".join(current_lines)
        kw_scores = rake.apply(text_chunk)
        for kw, score in kw_scores:
            aggregated_scores[kw] += score

    if not aggregated_scores:
        return None

    # Sort keywords by score
    sorted_keywords = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
    return doc_id, sorted_keywords


def create_csv_header(output_file: str, num_keywords: int):
    """Creates the output CSV file with the correct header."""
    header = ["document_id"]
    for i in range(1, num_keywords + 1):
        header.extend([f"keyword{i}", f"score{i}"])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)


def write_csv_row(output_file: str, doc_id: str, keywords: list, num_keywords: int):
    """Appends a result row to the CSV."""
    row = [doc_id]
    top_kws = keywords[:num_keywords]

    for kw, score in top_kws:
        row.extend([kw, f"{score:.2f}"])

    # Pad with empty strings
    missing = num_keywords - len(top_kws)
    row.extend(["", ""] * missing)

    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def sort_csv_file(csv_file: str):
    """
    Sorts the CSV file alphabetically by the first column (document_id).
    Uses pandas if available for speed, otherwise falls back to a memory-efficient native sort.
    """
    print("--- Sorting Output CSV ---")

    # Try using Pandas first (fastest)
    try:
        import pandas as pd
        print("Using Pandas for sorting...")
        df = pd.read_csv(csv_file)
        df.sort_values(by=df.columns[0], inplace=True)
        df.to_csv(csv_file, index=False)
        print("Sorting complete.")
        return
    except ImportError:
        print("Pandas not found. Using native Python sort (slower for massive files)...")

    # Native Python fallback (Memory efficient approach: read all, sort, write back)
    # WARNING: For strictly "millions" of rows, you might want an external merge sort.
    # Assuming the results file fits in RAM (usually fine for metadata even with 1M rows).

    temp_file = csv_file + ".tmp"

    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return  # Empty file

            # Read all rows into memory
            rows = list(reader)

        # Sort rows by first element (doc_id)
        rows.sort(key=lambda x: x[0])

        with open(temp_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        shutil.move(temp_file, csv_file)
        print("Sorting complete.")

    except Exception as e:
        print(f"Error during sorting: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)


def yield_document_tasks(input_dir: Path, lang: str, max_words: int, chunk_size: int):
    """Generator that identifies documents and yields task data."""
    # Use os.scandir for speed
    with os.scandir(input_dir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue

            dir_name = entry.name
            dir_path = Path(entry.path)

            # CASE A: Special "onepagers" folder
            if dir_name == "onepagers":
                file_groups = defaultdict(list)
                print(f"[Info] Scanning 'onepagers' directory: {dir_path}...")

                with os.scandir(dir_path) as file_entries:
                    for f_entry in file_entries:
                        if f_entry.is_file() and f_entry.name.lower().endswith(".txt"):
                            # Apply regex logic on the file stem (filename without extension)
                            file_stem = Path(f_entry.name).stem
                            match = ONEPAGER_DOC_REGEX.match(file_stem)

                            if match:
                                doc_id = match.group(1)
                            else:
                                # Fallback: use the whole stem as doc_id if pattern doesn't match
                                doc_id = file_stem

                            file_groups[doc_id].append(f_entry.path)

                for doc_id, files in file_groups.items():
                    yield (doc_id, files, lang, max_words, chunk_size)

            # CASE B: Standard Document Folder
            else:
                files = []
                with os.scandir(dir_path) as file_entries:
                    for f_entry in file_entries:
                        if f_entry.is_file() and f_entry.name.lower().endswith(".txt"):
                            files.append(f_entry.path)

                if files:
                    yield (dir_name, files, lang, max_words, chunk_size)

def main():
    parser = argparse.ArgumentParser(
        description="Extract keywords from massive archives of .txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input_dir", "-i", default="../PAGE_TXT", help="Root directory containing document folders.")
    parser.add_argument("--workers", "-j", type=int, default=max(1, multiprocessing.cpu_count() - 1),
                        help="Number of parallel worker processes.")
    parser.add_argument("--chunk_size", "-c", type=int, default=10000,
                        help="Number of lines to process in one memory batch per document.")
    parser.add_argument("--num_keywords", "-n", type=int, default=10,
                        help="Number of top keywords to save per document.")
    parser.add_argument("--lang", "-l", default="cs", help="Language code (e.g., 'cs', 'en').")
    parser.add_argument("--max_words", "-w", type=int, default=2, help="Maximum length (in words) of a keyword phrase.")
    parser.add_argument("--output_file", "-o", default="keywords_master.csv", help="Output CSV file path.")

    args = parser.parse_args()
    input_path = Path(args.input_dir)

    if not input_path.exists():
        print(f"Error: Directory '{input_path}' not found.")
        sys.exit(1)

    # Initialize CSV
    create_csv_header(args.output_file, args.num_keywords)

    print(f"--- Starting Processing ---")
    print(f"Input: {input_path.resolve()}")
    print(f"Workers: {args.workers}")

    processed_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        task_generator = yield_document_tasks(
            input_path, args.lang, args.max_words, args.chunk_size
        )

        for task in task_generator:
            future = executor.submit(process_document_task, task)
            futures[future] = task[0]

        print(f"--- Processing submitted tasks... ---")

        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                result = future.result()
                if result:
                    res_doc_id, keywords = result
                    write_csv_row(args.output_file, res_doc_id, keywords, args.num_keywords)
                    processed_count += 1

                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} documents...")
            except Exception as e:
                print(f"[Error] Failed processing document '{doc_id}': {e}")

    print(f"\n--- Processing Complete. Sorting Results... ---")
    sort_csv_file(args.output_file)

    print(f"--- Done! ---")
    print(f"Total documents processed: {processed_count}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()