import os
import csv
import argparse
from pathlib import Path
from multi_rake import Rake


def process_files_in_directory(directory_path: Path, lang: str, max_w_count: int):
    """
    Reads all .txt files from a *single* directory, combines their text,
    and extracts keywords using multi_rake.

    Returns:
        A list of (keyword, score) tuples, or None if an error occurs.
    """
    if not directory_path.is_dir():
        print(f"Error: Path '{directory_path}' is not a valid directory.")
        return None

    print(f"Processing .txt files in: {directory_path.resolve()}")

    # 1. Combine text from all .txt files in this directory
    all_text = ""
    file_count = 0
    # Use .glob("*.txt") to get files only in this directory, not subdirs
    for txt_file in directory_path.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"  # Add newline as a separator
                file_count += 1
        except Exception as e:
            print(f"Warning: Could not read file {txt_file.name}. Error: {e}")

    if not all_text:
        print("No .txt files found or files were empty.")
        return None

    print(f"Read content from {file_count} file(s).")

    # 2. Initialize Rake
    try:
        rake = Rake(language_code=lang, max_words=max_w_count)
    except Exception as e:
        print(f"Error initializing Rake (language_code='{lang}'). Is this language supported? Error: {e}")
        return None

    # 3. Apply Rake to the combined text
    print("Extracting keywords...")
    keywords_with_scores = rake.apply(all_text)

    if not keywords_with_scores:
        print("No keywords were extracted.")
        return None

    return keywords_with_scores


def run_recursive_processing(base_dir: str, num_keywords: int, lang: str, max_w_count: int, output_file: str):
    """
    Recursively processes subfolders within the base_dir.
    Each subfolder is processed individually, and the results
    are compiled into a single CSV file.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print(f"Error: Base path '{base_dir}' is not a valid directory.")
        return

    results_data = []

    # 1. Create CSV Header
    header = ["folder_name"]
    for i in range(1, num_keywords + 1):
        header.extend([f"keyword{i}", f"score{i}"])
    results_data.append(header)

    print(f"Starting recursive processing in: {base_path.resolve()}")

    # 2. Iterate through items in the base directory
    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            print(f"\n--- Processing Subfolder: {item.name} ---")

            # Get keywords for this subfolder
            keywords_with_scores = process_files_in_directory(item, lang, max_w_count)

            # Prepare the CSV row
            row = [item.name]
            if keywords_with_scores:
                keywords_to_add = keywords_with_scores[:num_keywords]

                for keyword, score in keywords_to_add:
                    row.extend([keyword, f"{score:.2f}"])

                # Pad the row if fewer than num_keywords were found
                num_missing_cols = (num_keywords - len(keywords_to_add)) * 2
                row.extend([""] * num_missing_cols)
            else:
                # Pad the row with empty values if no keywords found
                row.extend([""] * num_keywords * 2)

            results_data.append(row)

    # 3. Write results to CSV
    if len(results_data) <= 1:
        print("\nNo subfolders found or processed. CSV file will not be created.")
        return

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(results_data)
        print(f"\n--- Success! ---")
        print(f"Recursive processing complete. Results saved to: {output_file}")
    except Exception as e:
        print(f"\nError writing CSV file '{output_file}': {e}")


def main():
    """
    Main function to parse arguments and run the correct mode.
    """
    parser = argparse.ArgumentParser(
        description="Extract keywords from .txt files in a directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "input_dir",
        help="The input directory containing .txt files or subfolders to process.",
        default="page_texts"
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process each subfolder within 'input_dir' individually and save results to a CSV.\n"
             "If not set, processes all .txt files *directly* within 'input_dir' as one batch."
    )

    parser.add_argument(
        "-n", "--num_keywords",
        type=int,
        default=10,
        help="Number of top keywords to extract (default: 10)."
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
        help="Name of the output CSV file for recursive processing (default: 'keyword_results.csv')."
    )

    args = parser.parse_args()

    if args.recursive:
        # --- Recursive Mode (TODO 2 & 3) ---
        run_recursive_processing(
            args.input_dir,
            args.num_keywords,
            args.lang,
            args.max_words,
            args.output_file
        )
    else:
        # ---
        # ---
        print("Running in standard (non-recursive) mode.")
        keywords_with_scores = process_files_in_directory(
            Path(args.input_dir),
            args.lang,
            args.max_words
        )

        if keywords_with_scores:
            print(f"\n--- Top {args.num_keywords} Keywords for {args.input_dir} ---")
            for keyword, score in keywords_with_scores[:args.num_keywords]:
                print(f"{keyword:<30} (Score: {score:.2f})")
        else:
            print(f"No keywords found for {args.input_dir}.")


if __name__ == "__main__":
    main()