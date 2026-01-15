import sys
import os
import argparse
from pathlib import Path


def merge_single_pair(conllu_path, tsv_path, output_path):
    """
    Merges a single pair of CoNLL-U and TSV files.
    """
    tsv_data = []
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f_tsv:
            for line_num, line in enumerate(f_tsv, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                parts = line.split('\t')
                if len(parts) >= 2:
                    tsv_data.append({'token': parts[0], 'tag': parts[1], 'line': line_num})
                else:
                    # Fallback for lines with only a token (assume 'O' or '_')
                    tsv_data.append({'token': parts[0], 'tag': '_', 'line': line_num})
    except FileNotFoundError:
        print(f"Error: TSV file not found: {tsv_path}", file=sys.stderr)
        return False

    tsv_index = 0
    tsv_len = len(tsv_data)

    try:
        with open(conllu_path, 'r', encoding='utf-8') as f_conllu, \
                open(output_path, 'w', encoding='utf-8') as f_out:

            for line in f_conllu:
                stripped_line = line.strip()

                # Passthrough for comments and empty lines
                if not stripped_line or stripped_line.startswith('#'):
                    f_out.write(line)
                    continue

                cols = stripped_line.split('\t')

                # Check for valid ID column (skip multiword tokens like 1-2)
                if len(cols) >= 2 and '-' not in cols[0] and '.' not in cols[0]:
                    if tsv_index < tsv_len:
                        tsv_item = tsv_data[tsv_index]

                        # Warning for token mismatch
                        if cols[1] != tsv_item['token']:
                            print(f"[Warn] Token mismatch in {os.path.basename(conllu_path)}: "
                                  f"CoNLL-U '{cols[1]}' vs TSV '{tsv_item['token']}'", file=sys.stderr)

                        # Add tag to MISC column (column index 9)
                        new_attr = f"NER={tsv_item['tag']}"

                        if len(cols) > 9:
                            if cols[9] == '_':
                                cols[9] = new_attr
                            else:
                                cols[9] += f"|{new_attr}"
                        else:
                            # Fill missing columns if malformed
                            while len(cols) < 9:
                                cols.append('_')
                            cols.append(new_attr)

                        # Write updated line
                        f_out.write('\t'.join(cols) + '\n')
                        tsv_index += 1
                    else:
                        # TSV ended early
                        f_out.write(line)
                else:
                    # Write lines that aren't standard tokens as-is
                    f_out.write(line)

        print(f"[Info] Merged: {os.path.basename(conllu_path)} -> {output_path}")
        return True

    except FileNotFoundError:
        print(f"Error: CoNLL-U file not found: {conllu_path}", file=sys.stderr)
        return False


def process_directory(conllu_dir, tsv_dir, output_dir):
    """
    Scans conllu_dir for files and looks for matching .tsv files in tsv_dir.
    """
    conllu_path_obj = Path(conllu_dir)
    tsv_path_obj = Path(tsv_dir)
    output_path_obj = Path(output_dir)

    # Check inputs
    if not conllu_path_obj.exists():
        print(f"Error: CoNLL-U directory '{conllu_dir}' does not exist.")
        sys.exit(1)
    if not tsv_path_obj.exists():
        print(f"Error: TSV directory '{tsv_dir}' does not exist.")
        sys.exit(1)

    # Create output directory
    output_path_obj.mkdir(parents=True, exist_ok=True)

    conllu_files = list(conllu_path_obj.glob('*.conllu'))
    if not conllu_files:
        print(f"No .conllu files found in {conllu_dir}")
        return

    print(f"Found {len(conllu_files)} CoNLL-U files in '{conllu_dir}'. Matching with TSVs in '{tsv_dir}'...")

    for conllu_file in conllu_files:
        # Construct expected TSV path:
        # 1. Get filename stem (e.g. 'doc1' from 'doc1.conllu')
        # 2. Append .tsv
        # 3. Join with the TSV directory path
        tsv_file = tsv_path_obj / conllu_file.with_suffix('.tsv').name

        if tsv_file.exists():
            out_file = output_path_obj / f"{conllu_file.stem}_merged.conllu"
            merge_single_pair(conllu_file, tsv_file, out_file)
        else:
            print(f"[Skip] Missing matching TSV file: {tsv_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge CoNLL-U files with NER tags from TSV files.")

    # Output is always required
    parser.add_argument('-o', '--out', help="Output file path (single mode) or Output directory (batch mode)",
                        required=True)

    # Create two groups: Single Mode vs Batch Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)

    # Option 1: Batch Mode (Directories)
    mode_group.add_argument('--conllu-dir', help="Directory containing input CoNLL-U files")

    # Option 2: Single Mode (Files)
    mode_group.add_argument('--conllu-file', help="Single input CoNLL-U file")

    # Arguments for TSV locations (dependent on mode)
    parser.add_argument('--tsv-dir', help="Directory containing input TSV files (required if using --conllu-dir)")
    parser.add_argument('--tsv-file', help="Single input TSV file (required if using --conllu-file)")

    args = parser.parse_args()

    # Validate pairings
    if args.conllu_dir:
        if not args.tsv_dir:
            parser.error("If using --conllu-dir, you must also specify --tsv-dir.")
        process_directory(args.conllu_dir, args.tsv_dir, args.out)

    elif args.conllu_file:
        if not args.tsv_file:
            parser.error("If using --conllu-file, you must also specify --tsv-file.")
        merge_single_pair(args.conllu_file, args.tsv_file, args.out)


if __name__ == "__main__":
    main()
