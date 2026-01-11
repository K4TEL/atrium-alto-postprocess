#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ker.py

Purpose:
This script performs Keyword Extraction (KER) on a directory of text files
(e.g., ALTO XML or plain text). It is designed to work with models from
the 'ufal.morphodita' library for linguistic analysis (specifically, Czech
and English).

The script operates in two main phases:

1.  **Page-Level Extraction (Default):**
    - Scans an input directory for files.
    - Supports two directory structures:
        a) A directory of subdirectories (e.g., `input_dir/Book1/page1.xml`)
        b) A flat directory of files (e.g., `input_dir/Book1-page1.xml`)
    - For each file, it:
        - Extracts all text.
        - Normalizes the text using a Morphodita tagger (lemmatizes words,
          finds parts of speech, and filters for nouns/adjectives).
        - Calculates a TF-IDF score for each word. (TF = Term Frequency,
          IDF = Inverse Document Frequency).
        - Saves the top N keywords and their scores to a TSV file
          (defined by --out-tab).

2.  **Document-Level Summarization (Optional, with --sum):**
    - Reads the *page-level* TSV file created in Phase 1.
    - Aggregates (sums) the scores for all keywords belonging to the
      same document (e.g., "Book1").
    - Sorts the keywords by their total aggregated score.
    - Saves the top N keywords for the *entire document* to a separate
      summary TSV file (defined by --out-sum).

Dependencies:
- ufal.morphodita (Python library)
- python-magic (for file type detection)
- A pre-trained Morphodita .tagger model
- A pre-computed IDF .pickle table

Usage:
    # Phase 1: Create page-level keywords
    python run_ker.py --dir /path/to/alto_files --out-tab page_keywords.tsv --lang cs

    # Phase 2: Create document-level summary from the page-level file
    python run_ker.py --dir /path/to/alto_files --out-tab page_keywords.tsv --out-sum doc_keywords.tsv --lang cs --sum
"""

import math

try:
    # cPickle is a faster C-based implementation of pickle
    import cPickle as pickle
except:
    # Fallback to the standard Python implementation
    import pickle
import sys
import os
import re
import codecs
import argparse
import csv
import magic  # For detecting file types (e.g., XML, text, zip)
import xml.etree.ElementTree  # For parsing ALTO XML (used in text_util)
import zipfile  # For reading zip files (used in text_util)
from ufal.morphodita import Tagger, Forms, TaggedLemmas, TokenRanges
from pathlib import Path

# Import text extraction functions from the other utility file
# Note: This script re-implements some of these (lines_from_txt_file, etc.)
# This might be from an older version, but we'll comment what's here.
from text_util import *


def clean_lines(lines):
    """
    Cleans a list of text lines by removing common formatting marks,
    page numbers, and other noise using regular expressions.

    Args:
        lines (list[str]): A list of plain text lines.

    Yields:
        str: A cleaned line. If cleaning results in an empty string,
             it yields the original line to avoid losing content.
    """
    for line in lines:
        orginal = line.strip()
        line = line.strip()
        # Regex 1: Remove things like "  (continued)"
        line = re.sub("[[:space:]]+\([^(]*\)", "", line).strip()
        # Regex 2: Remove numbers and decimal numbers (like "1.2.3 ")
        line = re.sub(r"[0-9]+(\.[0-9]+)*\.?[[:space:]]*", "", line).strip()
        # Regex 3: Remove trailing dots and numbers (like page numbers " ... 123")
        line = re.sub(r"[[:space:]]*((\.+)|([[:space:]]+))[[:space:]]*[0-9]*$", "", line).strip()

        if line:
            yield line
        else:
            # If cleaning removed everything, return the original
            # This handles lines that *only* contain (e.g.) a page number
            yield orginal


class Morphodita(object):
    """
    A wrapper class to simplify interactions with the Morphodita tagger.

    This holds the loaded model and reusable objects to avoid
    re-initializing them for every line of text.
    """

    def __init__(self, model_file):
        """
        Instantiates Morphodita from a provided .tagger model file.

        Args:
            model_file (str): Path to the Morphodita .tagger model.
        """
        print(f"Loading Morphodita tagger from: {model_file}")
        self.tagger = Tagger.load(model_file)
        # These are re-usable objects required by the tagger API
        self.forms = Forms()
        self.lemmas = TaggedLemmas()
        self.tokens = TokenRanges()
        self.tokenizer = self.tagger.newTokenizer()
        print("Tagger loaded.")

    def normalize(self, text):
        """
        Processes a string of text to get a list of normalized "terms".

        Normalization involves:
        1. Tokenizing (splitting text into words)
        2. Tagging (finding Part-of-Speech, e.g., Noun, Adjective)
        3. Lemmatizing (finding the root form, e.g., "running" -> "run")
        4. Filtering (keeping only nouns/adjectives, removing stopwords)

        Args:
            text (str): The text to be processed.

        Returns:
            tuple (list[str], int):
                - A list of normalized, filtered, lowercase lemmas.
                - The total count of tokens (words) processed.
        """
        self.tokenizer.setText(text)
        lemmas = []
        token_count = 0
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            self.tagger.tag(self.forms, self.lemmas)
            token_count += len(self.lemmas)

            # This list comprehension is the core filtering logic
            lemmas += [l.lemma.lower() for l in self.lemmas \
                       # Keep only Nouns (NN) or Adjectives (AA)
                       if (l.tag == 'NN' or l.tag == 'AA') \
                       # Keep only words longer than 2 chars
                       and len(l.lemma) > 2 \
                       # Remove words in our hardcoded stopwords list
                       and l.lemma not in stopwords]
        return lemmas, token_count


def get_keywords(lines, tagger, idf_doc_count, idf_table, threshold, maximum_words):
    """
    Finds keywords in the provided lines of text using the TF-IDF measure.

    TF-IDF = Term Frequency * Inverse Document Frequency
    - TF: How often does a word appear in *this document*?
          (Words that appear often are important for this doc)
    - IDF: How rare is this word across *all documents*?
           (Rare words are more specific and thus more important)

    Args:
        lines (list[str]): Preprocessed lines of text for one page/doc.
        tagger (Morphodita): The initialized tagger object.
        idf_doc_count (int): Total number of documents in the corpus
                             (loaded from the IDF pickle).
        idf_table (dict): A pre-computed map of {word: doc_count}
                          (loaded from the IDF pickle).
        threshold (float): The minimum TF-IDF score to be considered a keyword.
        maximum_words (int): The max number of keywords to return.

    Returns:
        dict: A dictionary with 'keywords', 'keyword_scores', etc.
    """
    word_stat = {}  # This will store Term Frequency (TF) counts
    word_count = 0  # Total number of *normalized* words
    response = {}
    morphodita_calls = 0

    # --- 1. Calculate Term Frequency (TF) ---
    for line in clean_lines(lines):
        # Normalize the line (lemmatize, filter, etc.)
        norm_words, line_call_count = tagger.normalize(line)
        morphodita_calls += line_call_count
        for w in norm_words:
            if w not in word_stat:
                word_stat[w] = 0
            word_stat[w] += 1
            word_count += 1

    if word_count == 0:
        return {'keywords': [], 'keyword_scores': [], 'morphodita_calls': 0}

    word_count = float(word_count)

    # --- 2. Calculate TF-IDF for each word ---
    tf_idf = {}
    for word, count in word_stat.items():
        # Calculate TF (log-normalized frequency)
        tf = math.log(1 + count / word_count)

        # Calculate IDF (Inverse Document Frequency)
        idf = math.log(idf_doc_count)  # Default (high) score for unknown words
        if word in idf_table:
            # Word is known: log(Total Docs / Docs containing this word)
            idf = math.log(idf_doc_count / idf_table[word])

        # Final score
        tf_idf[word] = tf * idf

    # --- 3. Sort and Filter Keywords ---
    # Sort all words by their TF-IDF score, descending
    sorted_terms = sorted(word_stat.keys(), key=lambda x: -tf_idf[x])

    # Special logic:
    # 1. Always take the top 2 keywords, even if below the threshold.
    # 2. Then, take all other keywords *that are above* the threshold.
    # 3. Finally, limit the total list to `maximum_words`.
    keywords = (sorted_terms[:2] + [t for t in sorted_terms[2:] if tf_idf[t] >= threshold])[:maximum_words]

    response['keywords'] = keywords
    response['keyword_scores'] = [tf_idf[k] for k in keywords]
    response['morphodita_calls'] = morphodita_calls
    return response


def process_file(file_path, tagger, idf_doc_count, idf_table, threshold, maximum_words):
    """
    Takes a single file, detects its type, extracts text, and gets keywords.
    This function uses the text extraction functions defined in *this file*
    (lines_from_txt_file, etc.), which are duplicates of those in text_util.py.

    Args:
        (See get_keywords for most args)
        file_path (str): The full path to the file to process.

    Returns:
        tuple (dict, int):
            - A dictionary containing keywords or an error message.
            - An HTTP-like status code (200 for OK, 400 for error).
    """
    # Use python-magic to detect the file type
    file_info = magic.from_file(file_path)
    lines = []

    # --- 1. Detect File Type and Extract Text ---
    # These functions (lines_from_txt_file, etc.) are expected
    # to be defined in text_util.py, which was imported via `from text_util import *`
    if re.match("^UTF-8 Unicode (with BOM) text", file_info):
        lines = lines_from_txt_file(file_path, encoding='utf-8-sig')
    elif re.match("^UTF-8 Unicode", file_info):
        lines = lines_from_txt_file(file_path, encoding='utf-8')
    elif re.match("Unicode text, UTF-8 text", file_info):
        lines = lines_from_txt_file(file_path, encoding='utf-8')
    elif re.match("^ASCII text", file_info):
        lines = lines_from_txt_file(file_path, encoding='utf-8')
    elif re.match('^XML 1.0 document', file_info) and \
            (file_path.endswith('.alto') or file_path.endswith('.xml')):
        lines = lines_from_alto_file(file_path)
    elif re.match('^Zip archive data', file_info):
        lines = lines_from_zip_file(file_path)
    else:
        # Fallback if magic fails but the extension is correct
        if file_path.endswith('.txt'):
            lines = lines_from_txt_file(file_path, encoding='utf-8')
        elif file_path.endswith('.alto') or file_path.endswith('.xml'):
            lines = lines_from_alto_file(file_path)
        else:
            return {"error": "Unsupported file type: {}".format(file_info)}, 400

    if not lines:
        return {"error": "Empty file or no text extracted"}, 400

    # --- 2. Get Keywords ---
    return get_keywords(lines, tagger, idf_doc_count, idf_table, threshold, maximum_words), 200


def process_and_write_row(writer, file_path, tagger, idf_doc_count, idf_table, threshold, max_words, file_col, page_col,
                          lang_col):
    """
    A helper function that orchestrates the processing for a single file
    and writes the result as a row in the output CSV.

    Args:
        writer (csv.writer): The CSV writer object for the output file.
        file_path (str): Full path to the file to process.
        tagger (Morphodita): The loaded tagger.
        ... (other get_keywords args) ...
        max_words (int): Max keywords, used for padding.
        file_col (str): The 'file' ID for this row.
        page_col (str): The 'page' ID for this row.
        lang_col (str): The language code for this row.
    """
    # Start the row with the metadata
    row = [file_col, page_col, lang_col, threshold]
    try:
        # Call the main processing function
        data, code = process_file(file_path, tagger, idf_doc_count, idf_table, threshold, max_words)

        if code != 200:
            raise Exception(data.get("error", "Unknown processing error"))

        keywords = data.get('keywords', [])
        scores = data.get('keyword_scores', [])

        if len(keywords) == 0:
            # Don't write a row if no keywords were found
            # (Alternatively, could write a padded row)
            return

        # Add the keyword/score pairs to the row
        for k, s in zip(keywords, scores):
            row.extend([k, f"{s:.6f}"])  # Format score to 6 decimal places

        # --- Pad the row ---
        # All rows must have the same number of columns.
        # If we found fewer than max_words, fill the rest with empty strings.
        # (max_words - len(keywords)) is keywords_missing
        # * 2 because each keyword has a (keyword, score) pair.
        padding = (max_words - len(keywords)) * 2
        row.extend([""] * padding)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        # On error, pad the row with empty strings for all keyword/score columns
        padding = max_words * 2
        row.extend([""] * padding)

    # Write the complete (and padded) row to the CSV file
    writer.writerow(row)


# --- Main execution block ---
if __name__ == "__main__":

    # --- Define Default Paths ---
    # These are hardcoded paths, likely on a specific server.
    # They can be overridden by command-line arguments.
    taggers = {
        "cs": "/net/projects/morphodita/models/czech-morfflex-pdt-latest/czech-morfflex-pdt-161115-pos_only.tagger",
        "en": "/net/projects/morphodita/models/english-morphium-wsj-latest/english-morphium-wsj-140407.tagger"
    }

    idf_tables = {
        "cs": "ker_data/cs_idf_table.pickle",
        "en": "ker_data/en_idf_table.pickle"
    }

    # Hardcoded Czech stopwords
    stopwords = set((u"odstavec kapitola obsah část cvičení metoda druh rovnice" +
                     u"rejstřík literatura seznam základ příklad stanovení definice výpočet" +
                     u"csc prof ing doc" +
                     u"postup úvod poznámka závěr úloha zadání procvičení").split(" "))

    log_step = 100  # Print progress every 100 files

    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description='Runs KER keyword extraction on a directory.')
    parser.add_argument("--dir", help="Path to the input directory of files", required=True)
    parser.add_argument("--max-words", help="Maximum number N of keywords to extract (default: 10)", type=int,
                        default=10)
    parser.add_argument("--file-format", help="Input file format filter (default: alto)", choices=['txt', 'alto'],
                        default="alto")
    parser.add_argument("--lang", help="Language of the documents (default: cs)", choices=['cs', 'en'], default="cs")
    parser.add_argument("--out-tab",
                        help="Path to the output *page level* TSV table file (default: pages_keywords.tsv)",
                        default="pages_keywords.tsv")
    parser.add_argument("--out-sum",
                        help="Path to the output *document level* TSV table file (default: documents_keywords.tsv)",
                        default="documents_keywords.tsv")

    # Arguments to override the default model paths
    parser.add_argument("--cs-morphodita", help="Path to a Czech tagger model for Morphodita.", default=taggers['cs'])
    parser.add_argument("--cs-idf", help="Czech idf model.", default=idf_tables['cs'])
    parser.add_argument("--en-morphodita", help="Path to a English tagger model for Morphodita.", default=taggers['en'])
    parser.add_argument("--en-idf", help="English idf model.", default=idf_tables['en'])

    parser.add_argument("--threshold", help="Minimum tf-idf score for keywords (default: 0.2)", type=float, default=0.2)

    # This flag controls whether to run Phase 1 (page-level) or Phase 2 (doc-level)
    parser.add_argument("--sum", help="Summarize page-level keywords into document-level keywords", action='store_true')

    args = parser.parse_args()

    # --- 2. Load Models ---
    tagger = None
    idf_doc_count = None
    idf_table = None

    try:
        if args.lang == 'cs':
            if not os.path.exists(args.cs_morphodita):
                raise IOError(f"File with Czech Morphodita model does not exist: {args.cs_morphodita}")
            tagger = Morphodita(args.cs_morphodita)

            if not os.path.exists(args.cs_idf):
                raise IOError(f"File with Czech IDF model does not exist: {args.cs_idf}")
            # Load the pre-computed IDF table from the pickle file
            with open(args.cs_idf, 'rb') as f_idf:
                idf_doc_count = float(pickle.load(f_idf))  # First item is the total doc count
                idf_table = pickle.load(f_idf)  # Second item is the word:count map

        elif args.lang == 'en':
            # ... (same logic for English models) ...
            if not os.path.exists(args.en_morphodita):
                raise IOError(f"File with English Morphodita model does not exist: {args.en_morphodita}")
            tagger = Morphodita(args.en_morphodita)

            if not os.path.exists(args.en_idf):
                raise IOError(f"File with English IDF model does not exist: {args.en_idf}")
            with open(args.en_idf, 'rb') as f_idf:
                idf_doc_count = float(pickle.load(f_idf))
                idf_table = pickle.load(f_idf)

    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Models for language '{args.lang}' loaded.")

    # --- 3. Prepare Output CSV Header ---
    # Header will be: file, page, lang, threshold, keyword1, score1, keyword2, score2, ...
    header = ['file', 'page', 'lang', "threshold"]
    for i in range(1, args.max_words + 1):
        header.extend([f'keyword{i}', f'score{i}'])

    # Define a simple function to check file extensions
    if args.file_format == 'txt':
        file_check = lambda f: f.endswith('.txt')
    else:  # alto
        file_check = lambda f: f.endswith('.alto') or f.endswith('.alto.xml') or f.endswith('.xml')

    # --- 4. PHASE 1: Process Input Directory (if --sum is NOT set) ---
    if not args.sum:
        print(f"Starting Phase 1: Page-level keyword extraction.")
        print(f"Output will be written to: {args.out_tab}")
        try:
            # Open the output file for writing
            with open(args.out_tab, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out, delimiter='\t')
                writer.writerow(header)  # Write the header first

                input_dir = args.dir
                if not os.path.isdir(input_dir):
                    raise IOError(f"Input directory not found: {input_dir}")

                # --- 4a. Check Directory Structure ---
                # Check if the dir contains subdirs or is just a flat list of files
                has_subdirs = False
                for item in os.listdir(input_dir):
                    item_path = os.path.join(input_dir, item)
                    if os.path.isdir(item_path):
                        has_subdirs = True
                        break

                # --- 4b. Case 1: Directory contains subdirectories ---
                # e.g., input_dir/Book1/page1.txt
                # 'file' = "Book1", 'page' = "page1"
                if has_subdirs:
                    print("Processing directory with subdirectories...")
                    for isd, subdir in enumerate(sorted(os.listdir(input_dir))):
                        subdir_path = os.path.join(input_dir, subdir)
                        if not os.path.isdir(subdir_path):
                            continue

                        file_name_col = subdir  # The directory name is the 'file' ID
                        if isd % log_step == 0:
                            print(f"{isd}\tProcessing document: {file_name_col}")
                            f_out.flush()  # Save progress periodically

                        for file in sorted(os.listdir(subdir_path)):
                            if not file_check(file):
                                continue

                            file_path = os.path.join(subdir_path, file)
                            # Get page name, e.g., "book-001.alto.xml" -> "001"
                            page_name_col = Path(file).stem.split('.')[0].split('-')[-1]

                            # Process this single file
                            process_and_write_row(writer, file_path, tagger, idf_doc_count, idf_table,
                                                  args.threshold, args.max_words, file_name_col,
                                                  page_name_col, args.lang)

                # --- 4c. Case 2: Directory contains files (flat structure) ---
                # e.g., input_dir/Book1-001.txt
                # 'file' = "Book1", 'page' = "001"
                else:
                    print("Processing directory with files...")
                    for ifd, file in enumerate(sorted(os.listdir(input_dir))):
                        if not file_check(file):
                            continue

                        file_path = os.path.join(input_dir, file)
                        # Get stem, e.g., "Book1-001.alto.xml" -> "Book1-001"
                        file_stem = Path(file).stem.split('.')[0]

                        # Split on the *last* hyphen to separate file from page
                        parts = file_stem.rsplit('-', 1)

                        if len(parts) > 1:
                            file_name_col = parts[0]  # "Book1"
                            page_name_col = parts[1]  # "001"
                        else:
                            # No hyphen found, use whole stem as file and empty page
                            file_name_col = file_stem
                            page_name_col = ""

                        if ifd % log_step == 0:
                            print(f"{ifd}\tProcessing document: {file_name_col}")
                            f_out.flush()  # Save progress periodically

                        # Process this single file
                        process_and_write_row(writer, file_path, tagger, idf_doc_count, idf_table,
                                              args.threshold, args.max_words, file_name_col,
                                              page_name_col, args.lang)

            print(f"\nPhase 1 processing complete. Output table saved to: {args.out_tab}")
            # Set args.sum to True so that Phase 2 *always* runs after Phase 1
            args.sum = True

        except Exception as e:
            print(f"An error occurred during processing: {e}", file=sys.stderr)
            sys.exit(1)

    # --- 5. PHASE 2: Summarize Keywords (if --sum is set) ---
    if args.sum:
        print(f"\nStarting Phase 2: Document-level keyword summarization.")
        print(f"Reading from: {args.out_tab}")
        print(f"Output will be written to: {args.out_sum}")

        summary_output = args.out_sum
        try:
            # Open the page-level file (input) and summary file (output)
            with open(args.out_tab, 'r', encoding='utf-8') as f_in, \
                    open(summary_output, 'w', newline='', encoding='utf-8') as f_out:

                reader = csv.reader(f_in, delimiter='\t')
                writer = csv.writer(f_out, delimiter='\t')

                header = next(reader)
                keyword_cols = header[4:]  # Get all 'keywordN', 'scoreN' columns

                # Create the new summary header
                summary_header = ['file', 'lang', "threshold"] + keyword_cols
                writer.writerow(summary_header)

                current_file = None
                current_lang = None
                current_th = None
                # This dict will store {keyword: aggregated_score} for *one document*
                document_word_scores = {}

                print("Starting summary aggregation...")

                # --- 5a. Read the page-level CSV row by row ---
                for row in reader:
                    file_col = row[0]
                    lang_col = row[2]
                    threshold_col = row[3]

                    # Check if we are starting a new document
                    if current_file != file_col:

                        # --- 5b. Write the summary for the *previous* document ---
                        if current_file is not None:
                            # Sort all collected keywords by their total score
                            sorted_keywords = sorted(document_word_scores.items(), key=lambda item: item[1],
                                                     reverse=True)
                            # Get the top N
                            top_n_keywords_with_scores = sorted_keywords[:args.max_words]

                            # Build the summary row
                            summary_row = [current_file, current_lang, current_th]
                            for k, s in top_n_keywords_with_scores:
                                summary_row.extend([k, f"{s:.6f}"])  # Add (k, score)

                            # Pad the summary row
                            padding = (args.max_words - len(top_n_keywords_with_scores)) * 2
                            summary_row.extend([""] * padding)
                            writer.writerow(summary_row)
                        # --- End writing summary ---

                        # --- 5c. Reset for the new document ---
                        current_file = file_col
                        current_lang = lang_col
                        current_th = threshold_col
                        document_word_scores = {}

                    # --- 5d. Aggregate scores for the current document ---
                    # Iterate over the keyword/score pairs in this *page* row
                    for i in range(4, len(header), 2):
                        keyword = row[i]
                        score_str = row[i + 1]

                        if keyword and score_str:  # Ensure both exist
                            try:
                                score = float(score_str)
                                # Add this page's score to the document's total score
                                document_word_scores[keyword] = document_word_scores.get(keyword, 0.0) + score
                            except ValueError:
                                continue  # Skip if score is not a valid number

                # --- 5e. Write the summary for the *very last* document ---
                # (This code runs after the loop has finished)
                if current_file is not None:
                    print("Writing summary for last document:", current_file)
                    sorted_keywords = sorted(document_word_scores.items(), key=lambda item: item[1], reverse=True)
                    top_n_keywords_with_scores = sorted_keywords[:args.max_words]
                    summary_row = [current_file, current_lang, current_th]
                    for k, s in top_n_keywords_with_scores:
                        summary_row.extend([k, f"{s:.6f}"])
                    padding = (args.max_words - len(top_n_keywords_with_scores)) * 2
                    summary_row.extend([""] * padding)
                    writer.writerow(summary_row)
                # --- End writing last summary ---

            print(f"Phase 2 summary processing complete. Summary table saved to: {summary_output}")

        except Exception as e:
            print(f"An error occurred during summary processing: {e}", file=sys.stderr)
            sys.exit(1)