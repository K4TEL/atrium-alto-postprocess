#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

try:
    import cPickle as pickle
except:
    import pickle
import sys
import os
import re
import codecs
import argparse
import csv
import magic
import xml.etree.ElementTree
import zipfile
from ufal.morphodita import Tagger, Forms, TaggedLemmas, TokenRanges
from pathlib import Path

from text_util import *


def clean_lines(lines):
    """
    Returns the text that are present in a file after removing formating
    marks

    :param lines: List of plain text lines
    :type lines: list[str]
    """
    for line in lines:
        orginal = line.strip()
        line = line.strip()
        line = re.sub("[[:space:]]+\([^(]*\)", "", line).strip()
        line = re.sub(r"[0-9]+(\.[0-9]+)*\.?[[:space:]]*", "", line).strip()
        line = re.sub(r"[[:space:]]*((\.+)|([[:space:]]+))[[:space:]]*[0-9]*$", "", line).strip()
        if line:
            yield line
        else:
            yield orginal


class Morphodita(object):
    """
    A wrapper class for stuff needed fro working with Moprhodita.
    """

    def __init__(self, model_file):
        """
        Instantiates Morphodita from a provided model file.

        :param model_file: Path to the model file,
        :type model_file: str
        """
        self.tagger = Tagger.load(model_file)
        self.forms = Forms()
        self.lemmas = TaggedLemmas()
        self.tokens = TokenRanges()
        self.tokenizer = self.tagger.newTokenizer()

    def normalize(self, text):
        """
        Returns lematized nouns and adjectives from a provided text.

        :param text: Text to be processed
        :type text: str
        """
        self.tokenizer.setText(text)
        lemmas = []
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            self.tagger.tag(self.forms, self.lemmas)
            lemmas += [l.lemma.lower() for l in self.lemmas \
                       if (l.tag == 'NN' or l.tag == 'AA') and len(l.lemma) > 2 and l.lemma not in stopwords]
        return lemmas, len(self.lemmas)


def get_keywords(lines, tagger, idf_doc_count, idf_table, threshold, maximum_words):
    """
    Finds keywords in the provided lines of text using the tf-idf measure.

    :param lines: Preprocessed lines of text
    :type lines: list[str]

    :param tagger: Loaded Morphodita model for normalization of the text
    :type tagger: Morphodita

    :param idf_doc_count: Number of documents used for creating the idf table
    :type idf_doc_count: int

    :param idf_table: Precomputed IDF table.
    :type idf_table: dict

    :param threshold: Minimum score that is acceptable for a keyword.

    :param maximum_words: Maximum number of words to be returned.

    """
    word_stat = {}
    word_count = 0
    response = {}

    morphodita_calls = 0
    for line in clean_lines(lines):
        norm_words, line_call_count = tagger.normalize(line)
        morphodita_calls += line_call_count
        for w in norm_words:
            if w not in word_stat:
                word_stat[w] = 0
            word_stat[w] += 1
            word_count += 1
    word_count = float(word_count)

    tf_idf = {}
    for word, count in word_stat.items():
        tf = math.log(1 + count / word_count)
        idf = math.log(idf_doc_count)
        if word in idf_table:
            idf = math.log(idf_doc_count / idf_table[word])
        tf_idf[word] = tf * idf

    sorted_terms = sorted(word_stat.keys(), key=lambda x: -tf_idf[x])
    # Ensure at least 2 keywords are included (if possible) regardless of threshold, then filter by threshold
    keywords = (sorted_terms[:2] + [t for t in sorted_terms[2:] if tf_idf[t] >= threshold])[:maximum_words]
    response['keywords'] = keywords
    response['keyword_scores'] = [tf_idf[k] for k in keywords]
    response['morphodita_calls'] = morphodita_calls
    return response


def process_file(file_path, tagger, idf_doc_count, idf_table, threshold, maximum_words):
    """
    Takes the uploaded file, detecs its type (plain text, alto XML, zip)
    and calls a parsing function accordingly. If everything succeeds it
    returns keywords and 200 code, returns an error otherwise.
    """
    file_info = magic.from_file(file_path)
    lines = []
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
        # Fallback for .txt or .xml/.alto if magic fails
        if file_path.endswith('.txt'):
            lines = lines_from_txt_file(file_path, encoding='utf-8')
        elif file_path.endswith('.alto') or file_path.endswith('.xml'):
            lines = lines_from_alto_file(file_path)
        else:
            return {"error": "Unsupported file type: {}".format(file_info)}, 400

    if not lines:
        return {"error": "Empty file or no text extracted"}, 400

    # get_keywords is now defined in this same file
    return get_keywords(lines, tagger, idf_doc_count, idf_table, threshold, maximum_words), 200





def process_and_write_row(writer, file_path, tagger, idf_doc_count, idf_table, threshold, max_words, file_col, page_col,
                          lang_col):
    """
    Helper function to process a single file and write its keyword row to the output.
    """
    row = [file_col, page_col, lang_col, threshold]
    try:
        # process_file is now defined in this same file
        data, code = process_file(file_path, tagger, idf_doc_count, idf_table, threshold, max_words)

        if code != 200:
            raise Exception(data.get("error", "Unknown processing error"))

        keywords = data.get('keywords', [])
        scores = data.get('keyword_scores', [])

        if len(keywords) == 0:
            return

        for k, s in zip(keywords, scores):
            row.extend([k, f"{s:.6f}"])  # Format score to 6 decimal places

        # Pad row with empty strings if fewer than max_words keywords found
        padding = (max_words - len(keywords)) * 2
        row.extend([""] * padding)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        # Pad row with empty strings on error
        padding = max_words * 2
        row.extend([""] * padding)

    writer.writerow(row)


if __name__ == "__main__":
    # --- Define Default Paths ---
    # These paths are hardcoded but can be overridden by command-line args
    taggers = {
        "cs": "/net/projects/morphodita/models/czech-morfflex-pdt-latest/czech-morfflex-pdt-161115-pos_only.tagger",
        "en": "/net/projects/morphodita/models/english-morphium-wsj-latest/english-morphium-wsj-140407.tagger"
    }

    idf_tables = {
        "cs": "ker_data/cs_idf_table.pickle",
        "en": "ker_data/en_idf_table.pickle"
    }

    stopwords = set((u"odstavec kapitola obsah část cvičení metoda druh rovnice" +
                     u"rejstřík literatura seznam základ příklad stanovení definice výpočet" +
                     u"csc prof ing doc" +
                     u"postup úvod poznámka závěr úloha zadání procvičení").split(" "))

    log_step = 100

    parser = argparse.ArgumentParser(description='Runs KER keyword extraction on a directory.')

    # --- REQUIRED Argument ---
    parser.add_argument("--dir", help="Path to the input directory of files", required=True)

    # --- OPTIONAL Arguments (with defaults) ---
    parser.add_argument("--max-words", help="Maximum number N of keywords to extract (default: 10)", type=int,
                        default=10)
    parser.add_argument("--file-format", help="Input file format filter (default: alto)", choices=['txt', 'alto'],
                        default="alto")
    parser.add_argument("--lang", help="Language of the documents (default: cs)", choices=['cs', 'en'],
                        default="cs")
    parser.add_argument("--out-tab", help="Path to the output page level TSV table file (default: pages_keywords.tsv)",
                        default="pages_keywords.tsv")
    parser.add_argument("--out-sum", help="Path to the output document level TSV table file (default: documents_keywords.tsv)",
                        default="documents_keywords.tsv")

    # Optional model/idf paths (defaults loaded from dicts above)
    parser.add_argument("--cs-morphodita", help="Path to a Czech tagger model for Morphodita.",
                        default=taggers['cs'])
    parser.add_argument("--cs-idf", help="Czech idf model.",
                        default=idf_tables['cs'])
    parser.add_argument("--en-morphodita", help="Path to a English tagger model for Morphodita.",
                        default=taggers['en'])
    parser.add_argument("--en-idf", help="English idf model.",
                        default=idf_tables['en'])

    # Optional threshold
    parser.add_argument("--threshold", help="Minimum tf-idf score for keywords (default: 0.2)", type=float,
                        default=0.2)

    parser.add_argument("--sum", help="Summarize page-level keywords into document-level keywords", action='store_true')

    args = parser.parse_args()

    # --- 1. Load Models ---
    tagger = None
    idf_doc_count = None
    idf_table = None

    try:
        # **BUG FIX**: Changed args.language to args.lang
        if args.lang == 'cs':
            if not os.path.exists(args.cs_morphodita):
                raise IOError(f"File with Czech Morphodita model does not exist: {args.cs_morphodita}")
            tagger = Morphodita(args.cs_morphodita)

            if not os.path.exists(args.cs_idf):
                raise IOError(f"File with Czech IDF model does not exist: {args.cs_idf}")
            with open(args.cs_idf, 'rb') as f_idf:
                idf_doc_count = float(pickle.load(f_idf))
                idf_table = pickle.load(f_idf)

        elif args.lang == 'en':
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

    # --- 2. Prepare Output File ---
    header = ['file', 'page', 'lang', "threshold"]
    for i in range(1, args.max_words + 1):
        header.extend([f'keyword{i}', f'score{i}'])

    # Define file filter function
    if args.file_format == 'txt':
        file_check = lambda f: f.endswith('.txt')
    else:  # alto
        file_check = lambda f: f.endswith('.alto') or f.endswith('.alto.xml') or f.endswith('.xml')

    if not args.sum:
        # --- 3. Process Input Directory ---
        try:
            with open(args.out_tab, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out, delimiter='\t')
                writer.writerow(header)

                input_dir = args.dir

                # Check if directory contains subdirectories
                has_subdirs = False
                if not os.path.isdir(input_dir):
                    raise IOError(f"Input directory not found: {input_dir}")

                for item in os.listdir(input_dir):
                    item_path = os.path.join(input_dir, item)
                    if os.path.isdir(item_path):
                        has_subdirs = True
                        break

                # --- Case 1: Directory contains subdirectories ---
                # input_dir/Book1/page1.txt -> file=Book1, page=page1
                if has_subdirs:
                    print("Processing directory with subdirectories...")
                    for isd, subdir in enumerate(sorted(os.listdir(input_dir))):
                        subdir_path = os.path.join(input_dir, subdir)
                        if not os.path.isdir(subdir_path):
                            continue

                        file_name_col = subdir  # Directory name is the 'file'
                        if isd % log_step == 0:
                            print(f"{isd}\tProcessing document: {file_name_col}")
                            f_out.flush()

                        for file in sorted(os.listdir(subdir_path)):
                            if not file_check(file):
                                continue

                            file_path = os.path.join(subdir_path, file)
                            page_name_col = Path(file).stem.split('.')[0].split('-')[-1]

                            process_and_write_row(writer, file_path, tagger, idf_doc_count, idf_table,
                                                  args.threshold, args.max_words, file_name_col,
                                                  page_name_col, args.lang)

                # --- Case 2: Directory contains files ---
                # input_dir/File-001.txt -> file=File, page=001
                # input_dir/AnotherFile.txt -> file=AnotherFile, page=""
                else:
                    print("Processing directory with files...")
                    for ifd, file in enumerate(sorted(os.listdir(input_dir))):
                        if not file_check(file):
                            continue

                        file_path = os.path.join(input_dir, file)
                        file_stem = Path(file).stem.split('.')[0]

                        parts = file_stem.rsplit('-', 1)  # Split on the *last* hyphen

                        if len(parts) > 1:
                            file_name_col = parts[0]
                            page_name_col = parts[1]
                        else:
                            file_name_col = file_stem
                            page_name_col = ""

                        if ifd % log_step == 0:
                            print(f"{ifd}\tProcessing document: {file_name_col}")
                            f_out.flush()

                        process_and_write_row(writer, file_path, tagger, idf_doc_count, idf_table,
                                              args.threshold, args.max_words, file_name_col,
                                              page_name_col, args.lang)

            print(f"Processing complete. Output table saved to: {args.out_tab}")
            args.sum = True

        except Exception as e:
            print(f"An error occurred during processing: {e}", file=sys.stderr)
            sys.exit(1)

    if args.sum:
        # summarize page level keywords per file
        summary_output = args.out_sum
        try:
            with open(args.out_tab, 'r', encoding='utf-8') as f_in, \
                    open(summary_output, 'w', newline='', encoding='utf-8') as f_out:
                reader = csv.reader(f_in, delimiter='\t')
                writer = csv.writer(f_out, delimiter='\t')

                header = next(reader)
                keyword_cols = header[4:]  # All keyword and score columns

                # Create the new summary header (file, lang, keyword1, score1, ...)
                summary_header = ['file', 'lang', "threshold"] + keyword_cols
                writer.writerow(summary_header)

                current_file = None
                current_lang = None
                # This will store {keyword: aggregated_score}
                document_word_scores = {}

                print("Starting summary processing...")

                for row in reader:
                    file_col = row[0]
                    lang_col = row[2]
                    threshold_col = row[3]

                    if current_file != file_col:
                        # --- Write previous file's summary ---
                        if current_file is not None:
                            # Sort keywords by aggregated score, descending
                            sorted_keywords = sorted(document_word_scores.items(), key=lambda item: item[1],
                                                     reverse=True)
                            # Get top N
                            top_n_keywords_with_scores = sorted_keywords[:args.max_words]

                            summary_row = [current_file, current_lang, current_th]

                            for k, s in top_n_keywords_with_scores:
                                summary_row.extend([k, f"{s:.6f}"])  # Format score

                            # Pad row with empty strings
                            padding = (args.max_words - len(top_n_keywords_with_scores)) * 2
                            summary_row.extend([""] * padding)

                            writer.writerow(summary_row)
                        # --- End writing summary ---

                        # Reset for new file
                        current_file = file_col
                        current_lang = lang_col
                        current_th = threshold_col
                        document_word_scores = {}

                    # Aggregate keywords and scores for the current file
                    # Iterate over keyword/score pairs
                    for i in range(4, len(header), 2):
                        keyword = row[i]
                        score_str = row[i + 1]

                        if keyword and score_str:  # Ensure both exist
                            try:
                                score = float(score_str)
                                # Add the score to the keyword's total
                                document_word_scores[keyword] = document_word_scores.get(keyword, 0.0) + score
                            except ValueError:
                                # Handle case where score is not a valid float
                                continue

                                # --- Write LAST file's summary ---
                if current_file is not None:
                    # Sort keywords by aggregated score, descending
                    print("Writing summary for last document:", current_file)
                    sorted_keywords = sorted(document_word_scores.items(), key=lambda item: item[1], reverse=True)
                    # Get top N
                    top_n_keywords_with_scores = sorted_keywords[:args.max_words]

                    summary_row = [current_file, current_lang, current_th]

                    for k, s in top_n_keywords_with_scores:
                        summary_row.extend([k, f"{s:.6f}"])  # Format score

                    # Pad row with empty strings
                    padding = (args.max_words - len(top_n_keywords_with_scores)) * 2
                    summary_row.extend([""] * padding)

                    writer.writerow(summary_row)
                # --- End writing summary ---

            print(f"Summary processing complete. Summary table saved to: {summary_output}")

        except Exception as e:
            print(f"An error occurred during summary processing: {e}", file=sys.stderr)
            sys.exit(1)