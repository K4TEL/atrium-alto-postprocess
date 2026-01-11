import os
import csv
import argparse
import subprocess
import sys
from pathlib import Path
from multi_rake import Rake
from collections import defaultdict


def get_text_from_alto(xml_path: str) -> list[str]:
    """
    (Unchanged)
    Runs the 'alto-tools -t' command on a given ALTO XML file to extract
    all text lines.

    Args:
        xml_path: The file path to the ALTO XML.

    Returns:
        A list of strings, where each string is a stripped text line.
        Returns an empty list if the file is not found or 'alto-tools' fails.
    """
    if not os.path.exists(xml_path):
        print(f"[Warning] ALTO file not found: {xml_path}", file=sys.stderr)
        return []

    cmd = ["alto-tools", "-t", xml_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=True)
        
        # --- FIX START ---
        # Get all words as a single stream to handle line breaks issues
        raw_text = result.stdout
        words = raw_text.split()
        cleaned_words = []
        
        skip_next = False
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i+1]
            
            # Check if the current word is a prefix of the next word
            # (e.g., "staveb" is in "stavebreferát")
            # We also check length to ensure it's a real duplication pattern
            if len(next_word) > len(current_word) and next_word.startswith(current_word):
                # This is likely the prefix; skip adding it
                continue
            
            cleaned_words.append(current_word)
        
        # Don't forget the last word
        if words:
            cleaned_words.append(words[-1])
            
        # Reconstruct lines (or just return as list of strings if RAKE accepts it)
        # Since RAKE expects text, we can join them back.
        # However, your function returns list[str] (lines). 
        # For simplicity, we can return the cleaned text as one "line" per processed chunk 
        # or try to preserve original lines if possible. 
        # Given RAKE processes text blocks, joining is usually fine.
        
        return [" ".join(cleaned_words)]
    except subprocess.CalledProcessError as e:
        print(f"[Error] alto-tools failed on {xml_path}: {e.stderr}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("[Error] 'alto-tools' command not found.", file=sys.stderr)
        print("Please ensure 'alto-tools' is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Unexpected error processing {xml_path}: {e}", file=sys.stderr)
        return []
        
      


def process_directory_in_chunks(directory_path: Path, lang: str, max_w_count: int, chunk_size: int):
    """
    (MODIFIED)
    Reads all .xml files from a *single* directory, extracts text, and
    processes the text in chunks of 'chunk_size' lines to manage memory.

    Keywords from all chunks are aggregated and re-sorted to get the
    top keywords for the *entire* directory.

    Returns:
        A list of (keyword, score) tuples, or None if no text is found.
    """
    if not directory_path.is_dir():
        print(f"Error: Path '{directory_path}' is not a valid directory.")
        return None

    print(f"Processing .xml files in: {directory_path.resolve()}")

    try:
        rake = Rake(language_code=lang, max_words=max_w_count)
    except Exception as e:
        print(f"Error initializing Rake (language_code='{lang}'). Is this language supported? Error: {e}")
        return None

    current_lines = []
    # Use defaultdict to easily aggregate scores
    aggregated_keywords = defaultdict(float)
    total_files = 0
    total_lines = 0

    for xml_file in directory_path.glob("*.xml"):
        try:
            lines = get_text_from_alto(str(xml_file))
            if not lines:
                # Requirement: Ignore empty files
                print(f"Info: No text extracted from {xml_file.name}")
                continue

            current_lines.extend(lines)
            total_files += 1
            total_lines += len(lines)

            # Requirement: Process when chunk size is reached
            if len(current_lines) >= chunk_size:
                print(f"  ... processing chunk of {len(current_lines)} lines ...")
                text_chunk = "\n".join(current_lines)
                # Requirement: Don't save text permanently
                current_lines = []
                
                keywords_with_scores = rake.apply(text_chunk)
                
                # Aggregate keywords
                for kw, score in keywords_with_scores:
                    aggregated_keywords[kw] += score

        except Exception as e:
            print(f"Warning: Could not process file {xml_file.name}. Error: {e}")

    # Process any remaining lines after the loop
    if current_lines:
        print(f"  ... processing final chunk of {len(current_lines)} lines ...")
        text_chunk = "\n".join(current_lines)
        keywords_with_scores = rake.apply(text_chunk)
        for kw, score in keywords_with_scores:
            aggregated_keywords[kw] += score
    
    if not aggregated_keywords:
        print("No .xml files found or no text/keywords could be extracted.")
        return None

    print(f"Extracted text from {total_files} file(s) ({total_lines} lines total).")

    # Sort the aggregated keywords by score
    sorted_keywords = sorted(aggregated_keywords.items(), key=lambda item: item[1], reverse=True)

    print(f"\tExtracted top keywords:")
    for kw, score in sorted_keywords[:5]:
        print(f"\t - {kw} (Score: {score:.2f})")

    return sorted_keywords


def write_csv_row(output_file: str, row: list):
    """
    Helper function to append a single row to a CSV file.
    """
    try:
        # Requirement: Appending results
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"\nError appending to CSV file '{output_file}': {e}", file=sys.stderr)


def create_csv_header(output_file: str, num_keywords: int, is_recursive: bool):
    """
    Creates/overwrites the output file and writes the CSV header.
    """
    header = ["folder_name" if is_recursive else "chunk_name"]
    for i in range(1, num_keywords + 1):
        header.extend([f"keyword{i}", f"score{i}"])
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    except Exception as e:
        print(f"\nError writing CSV header to '{output_file}': {e}", file=sys.stderr)
        return False
    return True


def run_recursive_processing(base_dir: str, num_keywords: int, lang: str, max_w_count: int, output_file: str, chunk_size: int):
    """
    (MODIFIED)
    Recursively processes subfolders. Results are *appended* to the
    CSV file immediately after each subfolder is processed.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print(f"Error: Base path '{base_dir}' is not a valid directory.")
        return

    # Create CSV file with header first
    if not create_csv_header(output_file, num_keywords, is_recursive=True):
        return

    print(f"Starting recursive processing in: {base_path.resolve()}")
    folders_processed = 0

    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            print(f"\n--- Processing Subfolder: {item.name} ---")

            # Process this subfolder (which handles its own chunking)
            keywords_with_scores = process_directory_in_chunks(
                item, lang, max_w_count, chunk_size
            )

            # Requirement: empty files (folders) can be ignored entirely
            if not keywords_with_scores:
                print(f"No keywords found for {item.name}, skipping row.")
                continue

            # Prepare the CSV row
            row = [item.name]
            keywords_to_add = keywords_with_scores[:num_keywords]
            for keyword, score in keywords_to_add:
                row.extend([keyword, f"{score:.2f}"])

            # Pad the row if fewer than num_keywords were found
            num_missing_cols = (num_keywords - len(keywords_to_add)) * 2
            row.extend([""] * num_missing_cols)
            
            # Requirement: Appending results
            write_csv_row(output_file, row)
            folders_processed += 1

    print(f"\n--- Success! ---")
    if folders_processed > 0:
        print(f"Recursive processing complete. {folders_processed} folders saved to: {output_file}")
    else:
        print("No subfolders with keywords were found.")


def run_standard_processing(base_dir: str, num_keywords: int, lang: str, max_w_count: int, output_file: str, chunk_size: int):
    """
    (NEW FUNCTION)
    Processes all .xml files in a *single* directory in line-based chunks.
    Each chunk's results are saved as a *separate row* in the CSV.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print(f"Error: Base path '{base_dir}' is not a valid directory.")
        return
    
    # Create CSV file with header first
    if not create_csv_header(output_file, num_keywords, is_recursive=False):
        return
    
    print(f"Starting standard processing in: {base_path.resolve()}")

    try:
        rake = Rake(language_code=lang, max_words=max_w_count)
    except Exception as e:
        print(f"Error initializing Rake (language_code='{lang}'). Is this language supported? Error: {e}")
        return

    current_lines = []
    chunk_num = 1
    total_files = 0
    chunks_saved = 0

    for xml_file in sorted(base_path.glob("*.xml")):
        try:
            lines = get_text_from_alto(str(xml_file))
            if not lines:
                # Requirement: Ignore empty files
                continue
            
            current_lines.extend(lines)
            total_files += 1

            # Requirement: Process when chunk size is reached
            if len(current_lines) >= chunk_size:
                print(f"  ... processing chunk {chunk_num} ({len(current_lines)} lines) ...")
                text_chunk = "\n".join(current_lines)
                # Requirement: Don't save text permanently
                current_lines = []
                
                keywords_with_scores = rake.apply(text_chunk)
                
                if keywords_with_scores:
                    row = [f"chunk_{chunk_num}"]
                    keywords_to_add = keywords_with_scores[:num_keywords]
                    for keyword, score in keywords_to_add:
                        row.extend([keyword, f"{score:.2f}"])
                    
                    num_missing_cols = (num_keywords - len(keywords_to_add)) * 2
                    row.extend([""] * num_missing_cols)
                    
                    # Requirement: Appending results
                    write_csv_row(output_file, row)
                    chunks_saved += 1
                else:
                    print(f"  ... no keywords found for chunk {chunk_num}.")

                chunk_num += 1

        except Exception as e:
            print(f"Warning: Could not process file {xml_file.name}. Error: {e}")

    # Process any remaining lines as the final chunk
    if current_lines:
        print(f"  ... processing final chunk {chunk_num} ({len(current_lines)} lines) ...")
        text_chunk = "\n".join(current_lines)
        keywords_with_scores = rake.apply(text_chunk)
        
        if keywords_with_scores:
            row = [f"chunk_{chunk_num}"]
            keywords_to_add = keywords_with_scores[:num_keywords]
            for keyword, score in keywords_to_add:
                row.extend([keyword, f"{score:.2f}"])
            
            num_missing_cols = (num_keywords - len(keywords_to_add)) * 2
            row.extend([""] * num_missing_cols)
            
            write_csv_row(output_file, row)
            chunks_saved += 1
        else:
            print(f"  ... no keywords found for final chunk {chunk_num}.")
    
    print(f"\n--- Success! ---")
    if chunks_saved > 0:
        print(f"Standard processing complete. {total_files} files processed.")
        print(f"{chunks_saved} chunks saved to: {output_file}")
    else:
        print(f"{total_files} files processed, but no keywords were extracted.")


def main():
    """
    Main function to parse arguments and run the correct mode.
    (Help text updated)
    """
    parser = argparse.ArgumentParser(
        description="Extract keywords from ALTO .xml files.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "input_dir",
        help="The input directory containing ALTO .xml files or subfolders to process."
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process each subfolder within 'input_dir' individually.\n"
             "Results are aggregated per subfolder and saved as one row per folder.\n"
             "If not set, processes all .xml files *directly* within 'input_dir' as one batch,\n"
             "saving results *per chunk* of lines."
    )
    
    parser.add_argument(
        "-c", "--chunk_size",
        type=int,
        default=5000,
        help="Approximate number of text *lines* to process in one batch (default: 50000).\n"
             "In recursive mode, chunks are aggregated per folder.\n"
             "In standard mode, each chunk is saved as a new row."
    )

    parser.add_argument(
        "-n", "--num_keywords",
        type=int,
        default=10,
        help="Number of top keywords to extract per row (default: 10)."
    )

    parser.add_argument(
        "-l", "--lang",
        default="cs",
        help="Language code for stopword list (e.g., 'en', 'cs', 'de') (default: 'cs')."
    )

    parser.add_argument(
        "-w", "--max_words",
        type=int,
        default=2,
        help="Maximum number of words per keyword (default: 2)."
    )

    parser.add_argument(
        "-o", "--output_file",
        default="keyword_results.csv",
        help="Name of the output CSV file (default: 'keyword_results.csv')."
    )

    args = parser.parse_args()

    if args.recursive:
        # --- Recursive Mode ---
        run_recursive_processing(
            args.input_dir,
            args.num_keywords,
            args.lang,
            args.max_words,
            args.output_file,
            args.chunk_size
        )
    else:
        # --- Standard Mode (Non-Recursive) ---
        run_standard_processing(
            args.input_dir,
            args.num_keywords,
            args.lang,
            args.max_words,
            args.output_file,
            args.chunk_size
        )


if __name__ == "__main__":
    main()

