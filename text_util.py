#!/usr/bin/env python3
"""
text_util.py

Purpose:
This module provides a collection of utility functions for the ALTO
post-processing pipeline. It is not intended to be run directly, but
rather to be imported by other scripts (like run_langID.py).

Core functionalities include:
- Extracting text lines from ALTO XML files using 'alto-tools'.
- Calculating text "perplexity" using a transformer model (distilgpt2)
  to measure text quality (lower = more "normal" text).
- Autocorrecting text using language-specific spellcheckers.
- Classifying text lines into quality categories (Clear, Noisy, Trash, etc.)
  using a combination of language ID, perplexity, and heuristics.

This file contains both single-line and batch-optimized functions. The
batch functions (e.g., `calculate_perplexity_batch`, `classify_lines_batch`)
are preferred for performance.
"""

import pandas as pd
import fasttext
import sys
import re
import torch
from torch import nn
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
    all text lines. Also uses a simple caching strategy.

    If txt_path (a path to a .txt file) already exists, this function
    will *read from that .txt file* instead of re-processing the ALTO.

    If txt_path does not exist, it will:
    1. Run 'alto-tools -t' on the xml_path.
    2. Return the extracted lines.
    (Note: The calling script, run_langID.py, is responsible for *writing*
     the .txt file to complete the cache.)

    It also checks a backup path if the primary xml_path is not found.

    Args:
        xml_path: The file path to the ALTO XML.
        txt_path: The file path to the *target* .txt file (used for caching).

    Returns:
        A list of strings, where each string is a stripped text line.
        Returns an empty list if the file is not found or 'alto-tools' fails.
    """
    # --- Caching Logic ---
    if os.path.exists(txt_path):
        # If the text file already exists, just read from it.
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines

    # --- File Path Logic ---
    if not os.path.exists(xml_path):
        # If primary path fails, check a backup path
        backup_xml_path = Path(xml_path).parents[1] / "onepagers" / Path(xml_path).name
        if os.path.exists(backup_xml_path):
            xml_path = str(backup_xml_path)
        else:
            print(f"[Warning] ALTO file or its backup not found: {xml_path}", file=sys.stderr)
            return []

    # --- alto-tools Execution ---
    cmd = ["alto-tools", "-t", xml_path]
    try:
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
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Unexpected error processing {xml_path}: {e}", file=sys.stderr)
        return []


def autocorrect_text(input_text: str, lang_code: str, spellers: dict) -> tuple[str, list]:
    """
    Attempts to spell-correct a line of text using a language-specific speller.

    This function looks up the correct spell-checking library to use
    based on the `speller_per_language` config.

    Args:
        input_text: The text to correct.
        lang_code: The detected language code (e.g., "eng", "ces", "deu").
        spellers: A dictionary of pre-initialized speller objects.

    Returns:
        A tuple: (corrected_text: str, word_candidates: list)
    """

    speller, spell_type = None, 0

    # Find the correct speller to use for this language
    for lc, spell_setup in speller_per_language.items():
        if lang_code.startswith(lc):
            spell_type, spell_lang = spell_setup
            if spell_lang in spellers.keys():
                speller = spellers[spell_lang]
            else:
                # Initialize the speller if it's not already in the cache
                # (This is used in run_langID.py to pre-load common spellers)
                if spell_type == 1:
                    speller = Speller(spell_lang, only_replacements=True)
                else:
                    speller = SpellChecker(language=spell_lang)
            break

    if speller is None:
        # No speller configured for this language
        return input_text, []

    word_candidates = []
    input_words = input_text.split(" ")

    if spell_type == 1:
        # Use 'autocorrect' (Speller) for en, cs, etc.
        corrected = speller(input_text)
        # Note: This library doesn't easily provide *which* words were changed.
        for iw, word in enumerate(input_words):
            wc = speller.get_candidates(word)
            if len(wc) > 0:
                word_candidates.append((iw, wc))
    else:
        # Use 'pyspellchecker' (SpellChecker) for deu, etc.
        corrected_words = []
        for iw, word in enumerate(input_words):
            if word in speller:
                # Word is correct
                corrected_words.append(word)
            else:
                # Word is incorrect, find correction
                cor = speller.correction(word)
                wc = speller.candidates(word)
                if cor is None:
                    cor = word  # No correction found, use original
                corrected_words.append(cor)
                if wc is not None and len(wc) > 0:
                    word_candidates.append((iw, wc))
        corrected = " ".join(corrected_words)

    return corrected.strip(), word_candidates


def calculate_perplexity(text: str, model, tokenizer, device) -> float:
    """
    (SINGLE-LINE VERSION)
    Calculates the perplexity of a single line of text using the causal LM.

    Perplexity is a measure of how "surprising" a text is to the model.
    Low perplexity = "normal" text.
    High perplexity = "abnormal" text (e.g., OCR errors, gibberish).

    Args:
        text: The input text string.
        model: The pre-loaded AutoModelForCausalLM (e.g., distilgpt2).
        tokenizer: The pre-loaded AutoTokenizer.
        device: The device to run on ("cuda" or "cpu").

    Returns:
        The perplexity score as a float. Returns 0.0 for empty strings.
    """
    if not text:
        return 0.0  # No perplexity for empty strings

    try:
        # 1. Tokenize: Convert text to numerical IDs
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        if input_ids.size(1) == 0:
            return 0.0

        # 2. Truncate if text is too long for the model
        max_length = model.config.max_position_embeddings
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, :max_length]

        # 3. Set labels: For perplexity, labels are the same as inputs
        target_ids = input_ids.clone()

        # 4. Get model output
        with torch.no_grad():  # Disable gradient calculation to save memory
            outputs = model(input_ids, labels=target_ids)
            # The model handily returns the average loss
            neg_log_likelihood = outputs.loss

        # 5. Calculate perplexity: exp(loss)
        ppl = torch.exp(neg_log_likelihood)
        return ppl.item()

    except Exception as e:
        print(f"\n[Error] Perplexity calculation failed for text: '{text[:50]}...': {e}", file=sys.stderr)
        return 0.0  # Return 0.0 on error


# --- NEW: BATCH PERPLEXITY FUNCTION ---
def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    """
    (BATCH VERSION)
    Calculates the perplexity for a batch of text strings *at the same time*.
    This is much more efficient than calling `calculate_perplexity` in a loop.

    Args:
        texts (list[str]): A list of text strings to analyze.
        (model, tokenizer, device): Same as single-line version.

    Returns:
        list[float]: A list of perplexity scores, one for each input text.
    """
    if not texts:
        return []

    # --- 1. Filter out empty strings ---
    # We will process non-empty strings and fill in 0.0 for empty ones later.
    non_empty_texts = []
    original_indices = []  # To map results back to their original spots
    for i, text in enumerate(texts):
        if text and text.strip():
            non_empty_texts.append(text.strip())
            original_indices.append(i)

    if not non_empty_texts:
        return [0.0] * len(texts)  # All lines were empty

    # This tensor will hold the final results in the correct order
    results_tensor = torch.zeros(len(texts), device='cpu')

    try:
        # --- 2. Tokenize the entire batch ---
        max_length = model.config.max_position_embeddings
        tokenizer.pad_token = tokenizer.eos_token  # Use End-of-Sentence as padding
        encodings = tokenizer(
            non_empty_texts,
            return_tensors="pt",
            padding=True,  # Pad shorter texts to match the longest
            truncation=True,  # Truncate texts longer than max_length
            max_length=max_length
        )

        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        # --- 3. Create labels ---
        # Labels are inputs, but padding tokens are set to -100 to be ignored
        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = -100

        # --- 4. Get model output for the batch ---
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)

            # --- 5. Calculate Per-Sequence Loss ---
            # Unlike the single-line version, the batch 'outputs.loss' is the
            # *average* loss of the whole batch. We must calculate it per-line.

            logits = outputs.logits
            # Shift logits and labels for next-token-prediction loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            # Calculate loss for *every token* individually
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Reshape loss back to [batch_size, sequence_length]
            loss_per_token = loss.view(target_ids.size(0), -1)

            # Create a mask for non-padding tokens
            non_masked_tokens = (shift_labels != -100)

            # Sum the loss *for each sequence*
            sequence_loss = (loss_per_token * non_masked_tokens).sum(dim=1)
            # Count the number of non-padded tokens *for each sequence*
            num_tokens = non_masked_tokens.sum(dim=1)

            # Avoid division by zero
            num_tokens = torch.clamp(num_tokens, min=1)

            # 6. Calculate mean loss and perplexity for each sequence
            mean_loss = sequence_loss / num_tokens
            ppl_batch = torch.exp(mean_loss)

        # --- 7. Map results back to original order ---
        ppl_list = ppl_batch.to('cpu')
        for i, ppl_value in enumerate(ppl_list):
            results_tensor[original_indices[i]] = ppl_value.item()

        return results_tensor.tolist()

    except Exception as e:
        print(f"\n[Error] Batch perplexity calculation failed: {e}", file=sys.stderr)
        return [0.0] * len(texts)  # Fallback on error


# --- NEW: BATCH CLASSIFICATION FUNCTION ---
def classify_lines_batch(lines: list[str], model_ft, model_ppl, tokenizer_ppl, spellers: dict, device,
                         corrected_page_text="", corrected_page_lang="", corrected_page_lang_score=0.0,
                         corrected_page_perplexity=0.0) -> list[dict]:
    """
    (BATCH VERSION)
    Detects language, perplexity, and quality category for a batch of text lines.
    This is the core function of the quality assessment pipeline.

    Args:
        lines (list[str]): A list of raw text lines from a single page.
        model_ft: Pre-loaded fastText model.
        model_ppl: Pre-loaded perplexity model.
        tokenizer_ppl: Pre-loaded perplexity tokenizer.
        spellers: Dictionary of pre-loaded speller objects.
        device: "cuda" or "cpu".
        corrected_page_*: Page-level statistics used as context for "Short" lines.

    Returns:
        list[dict]: A list of result dictionaries, one for each input line.
    """
    n_lines = len(lines)
    if n_lines == 0:
        return []

    # This list will hold the final results in the correct order
    final_results = [None] * n_lines
    # This list will hold only the lines that are valid for processing
    lines_to_process = []
    original_indices = []  # Map from batch_index -> original_index

    # --- 1. Pre-filter Empty and Non-text lines ---
    # This loop quickly filters out lines that are clearly not text,
    # saving us from running expensive ML models on them.
    for i, line in enumerate(lines):
        clean_text = line.strip()

        # --- Handle Empty Line ---
        if not clean_text:
            final_results[i] = {
                "text": line, "lang_code": "N/A", "lang_score": 0.0,
                "perplexity": 0.0, "category": "Empty",
                "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
                "perplexity_corrected": 0.0
            }
            continue

        # --- Heuristic Pre-check (Quickly filter out non-text) ---
        n_chars = len(clean_text)
        letters = sum(c.isalpha() for c in clean_text)
        digits = sum(c.isdigit() for c in clean_text)
        symbols = sum(not c.isalnum() and not c.isspace() for c in clean_text)
        unique_symbols = set(c for c in clean_text if not c.isspace())

        if n_chars == 0:
            final_results[i] = {
                "text": line, "lang_code": "N/A", "lang_score": 0.0,
                "perplexity": 0.0, "category": "Empty",
                "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
                "perplexity_corrected": 0.0
            }
            continue  # Skip to next line

        letter_ratio = letters / n_chars
        digit_ratio = digits / n_chars
        symbol_ratio = symbols / n_chars

        # These rules define "Non-text":
        # - Less than 30% letters OR
        # - More than 40% digits OR
        # - More than 50% symbols OR
        # - Less than 4 characters OR
        # - Less than 3 unique non-space characters
        if (letter_ratio < 0.3 or digit_ratio > 0.4 or symbol_ratio > 0.5 or
                len(clean_text) < 4 or len(unique_symbols) < 3):
            final_results[i] = {
                "text": line, "lang_code": "N/A", "lang_score": 0.0,
                "perplexity": 0.0, "category": "Non-text",
                "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
                "perplexity_corrected": 0.0
            }
            continue  # Skip to next line

        # If the line passes all checks, add it to the batch to be processed
        lines_to_process.append(clean_text)
        original_indices.append(i)

    if not lines_to_process:
        return final_results  # All lines were empty or non-text

    # --- 3. Run Batch Predictions on Original Text ---
    # This is where the ML models run on all valid lines at once.
    try:
        # Get all perplexity scores
        ppl_results = calculate_perplexity_batch(lines_to_process, model_ppl, tokenizer_ppl, device)
        # Get all language predictions
        labels_batch, scores_batch = model_ft.predict(lines_to_process, k=1)

        lang_codes = [l[0].replace("__label__", "") for l in labels_batch]
        lang_scores = [s[0] for s in scores_batch]
    except Exception as e:
        print(f"\n[Error] Batch model prediction failed: {e}", file=sys.stderr)
        # On failure, mark all as "Trash"
        for i, batch_idx in enumerate(original_indices):
            final_results[batch_idx] = {
                "text": lines[batch_idx], "lang_code": "N/A", "lang_score": 0.0,
                "perplexity": 0.0, "category": "Trash",
                "corrected_text": "", "lang_corrected": "", "lang_score_corrected": 0.0,
                "perplexity_corrected": 0.0
            }
        return final_results

    # --- 4. & 5. Autocorrect and Run Predictions on Corrected Text ---

    # Identify lines that need correction (and are long enough to be worth it)
    texts_to_correct = []
    correction_batch_indices = []  # Index *within lines_to_process*

    for i, line in enumerate(lines_to_process):
        n_words = len(line.split(" "))
        # Only correct lines that are not "Short"
        if n_words >= 3:
            corrected, _ = autocorrect_text(line, lang_codes[i], spellers)
            if corrected != line:
                # Only run analysis if the text actually changed
                texts_to_correct.append(corrected)
                correction_batch_indices.append(i)

    # Run batch predictions on the *subset* of corrected texts
    ppl_corrected_results = []
    lang_codes_corrected = []
    lang_scores_corrected = []

    if texts_to_correct:
        try:
            ppl_corrected_results = calculate_perplexity_batch(texts_to_correct, model_ppl, tokenizer_ppl, device)
            labels_corr_batch, scores_corr_batch = model_ft.predict(texts_to_correct, k=1)
            lang_codes_corrected = [l[0].replace("__label__", "") for l in labels_corr_batch]
            lang_scores_corrected = [s[0] for s in scores_corr_batch]
        except Exception as e:
            print(f"\n[Error] Batch model prediction failed for corrected text: {e}", file=sys.stderr)
            texts_to_correct = []  # Clear this to prevent mapping

    # Map corrected results back to their original line index
    corrected_data_map = {}  # Map by batch_index
    for i, corrected_text in enumerate(texts_to_correct):
        batch_idx = correction_batch_indices[i]
        corrected_data_map[batch_idx] = {
            "corrected_text": corrected_text,
            "perplexity_corrected": ppl_corrected_results[i],
            "lang_corrected": lang_codes_corrected[i],
            "lang_score_corrected": lang_scores_corrected[i]
        }

    # --- 6. Categorize and Assemble Final Dictionaries ---
    for i, line in enumerate(lines_to_process):
        original_idx = original_indices[i]
        n_words = len(line.split(" "))

        # Get original text predictions
        text_perplexity = ppl_results[i]
        lang_code = lang_codes[i]
        score1 = lang_scores[i]

        # --- Handle "Short" lines ---
        # (Lines with < 3 words)
        if n_words < 3:
            final_results[original_idx] = {
                "text": lines[original_idx],  # Use original full line
                "lang_code": lang_code, "lang_score": score1,
                "perplexity": text_perplexity, "category": "Short",
                "corrected_text": "",  # Don't store corrected text for short lines
                # Use *page-level* stats as context
                "lang_corrected": corrected_page_lang,
                "lang_score_corrected": corrected_page_lang_score,
                "perplexity_corrected": corrected_page_perplexity
            }
            continue

        # --- Handle regular lines ---

        # Get corrected data if it exists for this line
        if i in corrected_data_map:
            corr_data = corrected_data_map[i]
            corrected_text = corr_data["corrected_text"]
            corrected_text_perplexity = corr_data["perplexity_corrected"]
            corrected_lang_code = corr_data["lang_corrected"]
            correcte_score1 = corr_data["lang_score_corrected"]
        else:
            # No correction was made or it failed
            corrected_text = ""
            corrected_text_perplexity = 0.0
            corrected_lang_code = ""
            correcte_score1 = 0.0

        # --- Categorize (Clear, Noisy, Trash, Rough) ---
        category = "Clear"  # Start with optimistic assumption
        letters = sum(c.isalpha() for c in line)
        upper_ratio = sum(c.isupper() for c in line) / letters if letters > 0 else 0.0
        is_common = lang_code.split("_")[0] in COMMON_LANGS
        is_cor_common = corrected_lang_code.split("_")[0] in COMMON_LANGS if corrected_lang_code else is_common
        # This 'is_Latin' check is commented out in the original, but was part of classify_line
        # is_Latin = lang_code.split("_")[-1] == "Latn"
        # is_cor_Latin = corrected_lang_code.split("_")[-1] == "Latn" if corrected_lang_code else is_Latin

        # --- Categorization Logic Tree ---

        # 1. Check for TRASH
        if text_perplexity >= PERPLEXITY_THRESHOLD_MAX:
            if corrected_text_perplexity == 0.0 or corrected_text_perplexity >= PERPLEXITY_THRESHOLD_MAX:
                category = "Trash"
        elif upper_ratio > 0.9 and letters > 10:  # Almost all-caps
            category = "Trash"
        # elif not is_Latin and not is_cor_Latin: # Not a latin script
        #     category = "Trash"

        # 2. Check for NOISY
        elif text_perplexity >= PERPLEXITY_THRESHOLD_MIN:
            if corrected_text_perplexity == 0.0 or corrected_text_perplexity >= PERPLEXITY_THRESHOLD_MIN:
                category = "Noisy"
        elif upper_ratio > 0.6 and letters > 10:  # Mostly all-caps
            category = "Noisy"
        elif not is_common and not is_cor_common:  # Not a common language
            category = "Noisy"
        elif score1 < LANG_SCORE_ROUGH and correcte_score1 < LANG_SCORE_ROUGH:  # Low confidence
            category = "Noisy"

        # 3. Check for ROUGH (promotion from Noisy)
        if correcte_score1 > LANG_SCORE_ROUGH and any(corrected_lang_code.startswith(cl) for cl in COMMON_LANGS):
            category = "Rough"
        elif score1 > LANG_SCORE_ROUGH and any(lang_code.startswith(cl) for cl in COMMON_LANGS):
            category = "Rough"

        # 4. Check for CLEAR (promotion from Rough)
        if score1 > LANG_SCORE_CLEAR and any(lang_code.startswith(cl) for cl in COMMON_LANGS):
            category = "Clear"

        # --- Assemble the final dictionary for this line ---
        final_results[original_idx] = {
            "text": lines[original_idx],
            "corrected_text": corrected_text,
            "lang_code": lang_code,
            "lang_corrected": corrected_lang_code,
            "lang_score": score1,
            "lang_score_corrected": correcte_score1,
            "perplexity": text_perplexity,
            "perplexity_corrected": corrected_text_perplexity,
            "category": category
        }

    return final_results



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
