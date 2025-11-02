#!/usr/bin/env python3
import pandas as pd
import fasttext
import sys
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import os
import csv
from pathlib import Path
from autocorrect import Speller
from spellchecker import SpellChecker

# --- Configuration ---
COMMON_LANGS = ["ces", "deu", "eng"]  # Languages considered "common"

# Perplexity Thresholds for 'distilgpt2'
# Perplexity measures how well a model "predicts" a text.
# Lower perplexity = more "normal" or "expected" text.
# Higher perplexity = more "surprising" or "abnormal" text (e.g., OCR errors, non-text).
PERPLEXITY_THRESHOLD_MAX = 500  # Perplexity >= this -> likely "Trash"
PERPLEXITY_THRESHOLD_MIN = 100  # Perplexity >= this -> likely "Noisy"

# Language Score Thresholds
# fastText language identification returns a score between 0 and 1.
# Higher score = more confidence in the predicted language.
LANG_SCORE_ROUGH = 0.5  # Language score >= this -> "Rough"
LANG_SCORE_CLEAR = 0.9  # Language score >= this -> "Clear"

# Spell-checkers for correction heuristics
speller_per_language = {
        "eng": (1, "en"),  # 1 or 0
        "ces": (1, "cs"),
        "deu": (0, "de"),
        # "pol": (1, "pl"),
        "rus": (1, "ru"),
        "ukr": (1, "uk"),
        # "tur": (1, "tr"),
        "spa": (0, "es"),
        "por": (0, "pt"),
        "ita": (0, "it"),
        "fra": (0, "fr"),
        "ell": (1, "el"),
        # "vie": (1, "vi"),
        "pes": (0, "fa"),
        "dan": (0, "nl"),
        "lvs": (0, "lv"),
        "eus": (0, "eu")
    }

def get_text_from_alto(xml_path: str, txt_path: str) -> list[str]:
    """
    Runs the 'alto-tools -t' command on a given ALTO XML file to extract
    all text lines.

    Args:
        xml_path: The file path to the ALTO XML.

    Returns:
        A list of strings, where each string is a stripped text line.
        Returns an empty list if the file is not found or 'alto-tools' fails.
    """
    if not os.path.exists(xml_path):
        backup_xml_path = Path(xml_path).parents[1] / "onepagers" / Path(xml_path).name
        if os.path.exists(backup_xml_path):
            xml_path = str(backup_xml_path)
        else:
            print(f"[Warning] ALTO file or its backup not found: {xml_path}", file=sys.stderr)
            return []

    if os.path.exists(txt_path):
        # print(f"[ALTO] Using recorded text file: {txt_path}", file=sys.stderr)
        # If the text file already exists, read from it
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines

    cmd = ["alto-tools", "-t", xml_path]
    try:
        # print(f"[ALTO] Extracting text from xml file: {xml_path}", file=sys.stderr)
        # Run alto-tools to extract text
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=True)
        # Split text into lines, strip whitespace, and filter out empty lines
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines
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


def autocorrect_text(input_text: str, lang_code: str, spellers: dict) -> tuple[str, list]:
    """
    Attempts to spell-correct a line of text using a language-specific speller.

    Note: The 'pyenchant' or 'autocorrect' libraries have different APIs.
    The German 'spellchecker' logic is different from 'autocorrect' (en, cs).

    Args:
        input_text: The text to correct.
        lang_code: The detected language code (e.g., "eng", "ces", "deu").

    Returns:
        A tuple:
        (corrected_text: str, word_candidates: list)
        where word_candidates is a list of (word_index, [candidates_list])
    """

    speller, spell_type = None, 0
    for lc, spell_setup in speller_per_language.items():
        if lang_code.startswith(lc):
            spell_type, spell_lang = spell_setup
            if spell_lang in spellers.keys():
                speller = spellers[spell_lang]
            else:
                if spell_type == 1:
                    speller = Speller(spell_lang, only_replacements=True)
                else:
                    speller = SpellChecker(language=spell_lang)
            break

    if speller is None:
        # No speller for this language
        return input_text, []


    word_candidates = []
    input_words = input_text.split(" ")

    if spell_type == 1:
        # Use 'autocorrect' (Speller) for en, cs
        corrected = speller(input_text)
        for iw, word in enumerate(input_words):
            wc = speller.get_candidates(word)
            if len(wc) > 0:
                word_candidates.append((iw, wc))
    else:
        # Use 'pyspellchecker' (SpellChecker) for deu
        corrected_words = []
        for iw, word in enumerate(input_words):
            # pyspellchecker is case-sensitive, so we check lower
            # but preserve original case if it's correct
            if word in speller:
                corrected_words.append(word)
            else:
                cor = speller.correction(word)
                wc = speller.candidates(word)
                if cor is None:
                    cor = word  # No correction found
                corrected_words.append(cor)
                if wc is not None and len(wc) > 0:
                    word_candidates.append((iw, wc))
        corrected = " ".join(corrected_words)

    return corrected.strip(), word_candidates


def calculate_perplexity(text: str, model, tokenizer, device) -> float:
    """
    Calculates the perplexity of a single line of text using the causal LM.

    Perplexity is calculated as exp(cross-entropy_loss).
    A high value indicates the text is "surprising" or "unlikely"
    (e.g., OCR errors, jumbled text).
    A low value indicates the text is "normal" or "expected" by the model.

    Args:
        text: The input text string.
        model: The pre-loaded AutoModelForCausalLM.
        tokenizer: The pre-loaded AutoTokenizer.

    Returns:
        The perplexity score as a float. Returns 0.0 for empty strings or
        in case of an error.
    """
    if not text:
        return 0.0  # No perplexity for empty strings

    try:
        # Tokenize the text
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        if input_ids.size(1) == 0:
            return 0.0  # Handle cases where text is just whitespace

        # The maximum sequence length the model can handle
        max_length = model.config.max_position_embeddings

        # If text is too long, truncate it
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, :max_length]

        # For a causal LM, labels are the same as input_ids
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Loss is the average negative log likelihood
            neg_log_likelihood = outputs.loss

        # Perplexity is the exponentiation of the loss
        ppl = torch.exp(neg_log_likelihood)
        return ppl.item()

    except Exception as e:
        print(f"\n[Error] Perplexity calculation failed for text: '{text[:50]}...': {e}", file=sys.stderr)
        return 0.0  # Return 0.0 on error


def classify_line(line: str, model_ft, model_ppl, tokenizer_ppl, spellers: dict, device,
                  corrected_page_text="", corrected_page_lang="", corrected_page_lang_score=0.0,
                  corrected_page_perplexity=0.0) -> dict:
    """
    Detects language, perplexity, and quality category for a single line of text.

    Args:
        line: The raw text line to classify.
        model_ft: The pre-loaded fastText model.
        model_ppl: The pre-loaded causal LM.
        tokenizer_ppl: The pre-loaded tokenizer.
        corrected_page_text (str): The autocorrected text for the *entire page*.
        corrected_page_lang (str): The language code for the *entire page*.
        corrected_page_lang_score (float): The language score for the *entire page*.
        corrected_page_perplexity (float): The perplexity for the *entire page*.
            **Note**: The 'corrected_page_*' args are used as fallback data for lines
            categorized as "Short", which are too short to classify reliably
            on their own.

    Returns:
        A dictionary containing all classification results for the line.
    """
    clean_text = line.strip()

    # --- 1. Handle Empty Line ---
    if not clean_text:
        return {
            "text": line, "lang_code": "N/A", "lang_score": 0.0,
            "perplexity": 0.0, "category": "Empty",
            "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
            "perplexity_corrected": 0.0
        }

    # --- 2. Heuristic Pre-check (Quickly filter out non-text) ---
    n_chars = len(clean_text)
    n_words = len(clean_text.split(" "))
    letters = sum(c.isalpha() for c in clean_text)
    digits = sum(c.isdigit() for c in clean_text)
    symbols = sum(not c.isalnum() and not c.isspace() for c in clean_text)
    unique_symbols = set(c for c in clean_text)
    space_chars = sum(c.isspace() for c in clean_text)

    # Avoid division by zero if n_chars is somehow 0
    if n_chars == 0 or space_chars == n_chars:
        return {
            "text": line, "lang_code": "N/A", "lang_score": 0.0,
            "perplexity": 0.0, "category": "Empty",
            "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
            "perplexity_corrected": 0.0
        }

    letter_ratio = letters / n_chars
    digit_ratio = digits / n_chars
    symbol_ratio = symbols / n_chars

    # These heuristics identify lines that are very unlikely to be prose.
    if (letter_ratio < 0.3 or  # Less than 30% letters
            digit_ratio > 0.4 or  # More than 40% digits
            symbol_ratio > 0.5 or  # More than 50% symbols
            len(clean_text) < 4 or  # Less than 4 characters
            len(unique_symbols) < 3):  # Less than 3 unique non-space characters
        return {
            "text": line, "lang_code": "N/A", "lang_score": 0.0,
            "perplexity": 0.0, "category": "Non-text",
            "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
            "perplexity_corrected": 0.0
        }

    # --- 3. Run Predictions on Original Text ---
    try:
        # Perplexity
        text_perplexity = calculate_perplexity(clean_text, model_ppl, tokenizer_ppl, device)

        # Language prediction
        labels, scores = model_ft.predict(clean_text, k=1)
        lang_code = labels[0].replace("__label__", "")
        score1 = float(scores[0])

    except Exception as e:
        print(f"\n[Error] Model prediction failed for line: '{line[:50]}...': {e}", file=sys.stderr)
        return {
            "text": line, "lang_code": "N/A", "lang_score": 0.0,
            "perplexity": 0.0, "category": "Trash",  # Fail to "Trash"
            "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
            "perplexity_corrected": 0.0
        }

    # --- 4. Handle "Short" lines ---
    # Lines with < 3 words are too short for reliable categorization.
    # We assign them the 'Short' category and backfill the 'corrected'
    # fields with the stats from the *entire page* for context.
    if n_words < 3:
        return {
            "text": line, "lang_code": lang_code, "lang_score": score1,
            "perplexity": text_perplexity, "category": "Short",
            # "corrected_text": corrected_page_text,  # Use page-level context
            "corrected_text": "",  # empty to save space
            "lang_corrected": corrected_page_lang,
            "lang_score_corrected": corrected_page_lang_score,
            "perplexity_corrected": corrected_page_perplexity
        }

    # --- 5. Run Predictions on Corrected Text (for comparison) ---
    corrected_text, candidates = autocorrect_text(clean_text, lang_code, spellers)

    if corrected_text == clean_text:
        # No corrections were made
        corrected_text = ""
        corrected_text_perplexity = 0.0
        corrected_lang_code = ""
        correcte_score1 = 0.0
    else:
        # Corrections were made, so let's analyze the corrected text
        try:
            corrected_text_perplexity = calculate_perplexity(corrected_text, model_ppl, tokenizer_ppl, device)
            labels, scores = model_ft.predict(corrected_text, k=1)  # Note: predicting on *corrected* text
            corrected_lang_code = labels[0].replace("__label__", "")
            correcte_score1 = float(scores[0])

        except Exception as e:
            print(f"\n[Error] Model prediction failed for corrected line: '{line[:50]}...': {e}", file=sys.stderr)
            # If correction analysis fails, just return original results and "Trash"
            return {
                "text": line, "lang_code": lang_code, "lang_score": score1,
                "perplexity": text_perplexity, "category": "Trash",
                "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
                "perplexity_corrected": 0.0
            }

    # --- 6. Categorize (Clear, Noisy, Trash, Rough) ---
    category = "Clear"  # Start with optimistic assumption

    # Gather features for categorization
    upper_ratio = sum(c.isupper() for c in clean_text) / letters if letters > 0 else 0.0
    is_Latin = lang_code.split("_")[-1] == "Latn"
    is_common = lang_code.split("_")[0] in COMMON_LANGS
    is_cor_Latin = corrected_lang_code.split("_")[-1] == "Latn" if corrected_lang_code else is_Latin
    is_cor_common = corrected_lang_code.split("_")[0] in COMMON_LANGS if corrected_lang_code else is_common

    # Check for TRASH (highest priority)
    if text_perplexity >= PERPLEXITY_THRESHOLD_MAX:
        if corrected_text_perplexity != 0 and corrected_text_perplexity >= PERPLEXITY_THRESHOLD_MAX:
            category = "Trash"  # Extremely high perplexity
    elif upper_ratio > 0.9 and letters > 10:
        category = "Trash"  # All caps (likely headers, not prose)
    elif not is_Latin and not is_cor_Latin:
        category = "Trash"  # Not a Latin script (e.g., Cyrillic, Greek)

    # Check for NOISY
    elif text_perplexity >= PERPLEXITY_THRESHOLD_MIN:
        if corrected_text_perplexity != 0 and  corrected_text_perplexity >= PERPLEXITY_THRESHOLD_MIN:
            category = "Noisy"  # High perplexity, and correction makes it *worse*
    elif upper_ratio > 0.6 and letters > 10:
        category = "Noisy"  # Mostly caps
    elif not is_common and not is_cor_common:
        category = "Noisy"  # Not one of our common languages
    elif score1 < LANG_SCORE_ROUGH and correcte_score1 < LANG_SCORE_ROUGH:
        category = "Noisy"  # Low confidence in *both* original and corrected

    # Check for ROUGH (better than Noisy, not as good as Clear)
    if correcte_score1 > LANG_SCORE_ROUGH and any(corrected_lang_code.startswith(cl) for cl in COMMON_LANGS):
        category = "Rough"  # Correction results in a decent, common lang
    elif score1 > LANG_SCORE_ROUGH and any(lang_code.startswith(cl) for cl in COMMON_LANGS):
        category = "Rough"  # Original is a decent, common lang

    # Check for CLEAR (highest quality)
    if score1 > LANG_SCORE_CLEAR and any(lang_code.startswith(cl) for cl in COMMON_LANGS):
        category = "Clear"  # High-confidence, common language

    return {
        "text": line,
        "corrected_text": corrected_text,
        "lang_code": lang_code,
        "lang_corrected": corrected_lang_code,
        "lang_score": score1,
        "lang_score_corrected": correcte_score1,
        "perplexity": text_perplexity,
        "perplexity_corrected": corrected_text_perplexity,
        "category": category
    }




# KER

def lines_from_txt_file(file_path, encoding='utf-8'):
    """
    Loads lines of text from a plain text file.

    :param file_path: Path to the alto file or a file-like object.

    """
    if type(file_path) is str:
        f = codecs.open(file_path, 'r', encoding)
    else:
        f = codecs.getreader(encoding)(file_path)

    content = [l.strip() for l in f]
    f.close()
    return content


def lines_from_alto_file(file_path):
    """
    Loads lines of text from a provided alto file.

    :param file_path: Path to the alto file or a file-like object.

    """
    try:
        e = xml.etree.ElementTree.parse(file_path).getroot()
    except xml.etree.ElementTree.ParseError as pe:
        raise Exception("XML ParseError in {}: {}".format(file_path, pe))

    layout = None

    # Support for ALTO with and without namespaces
    namespace = ''
    if '}' in e.tag:
        namespace = e.tag.split('}')[0] + '}'

    for c in e:
        if c.tag.endswith('Layout'):
            layout = c
            break
    if layout is None:
        raise Exception("XML is not ALTO file (does not contain layout object).")

    text_lines = layout.findall(".//{}TextLine".format(namespace))

    for text_line in text_lines:
        line_words = []
        for string in text_line:
            if not string.tag.endswith('String'):
                continue
            if 'CONTENT' in string.attrib:
                line_words.append(string.attrib['CONTENT'])
        yield " ".join(line_words)


def lines_from_zip_file(file_path):
    """
    Loads lines of text from a provided zip file. If it contains alto file, it
    uses them, otherwise looks for txt files. Files can in an arbitrary depth.

    :param file_path: Path to the uploaded zip file.
    :type file_path: str

    """
    archive = zipfile.ZipFile(file_path)
    alto_files = [n for n in archive.namelist() if n.endswith(".alto") or n.endswith(".xml")]
    if alto_files:
        for f_name in alto_files:
            if f_name.startswith('__MACOSX') or f_name.endswith('/'): continue
            try:
                with archive.open(f_name) as f:
                    for line in lines_from_alto_file(f):
                        yield line
            except Exception as e:
                print("Error processing {} in zip: {}".format(f_name, e), file=sys.stderr)
    else:
        txt_files = [n for n in archive.namelist() if n.endswith(".txt")]
        if not txt_files:
            raise Exception("Archive contains neither alto files nor text files.")
        for f_name in txt_files:
            if f_name.startswith('__MACOSX') or f_name.endswith('/'): continue
            try:
                with archive.open(f_name) as f:
                    for line in lines_from_txt_file(f):
                        yield line
            except Exception as e:
                print("Error processing {} in zip: {}".format(f_name, e), file=sys.stderr)
