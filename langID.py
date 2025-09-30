import pandas as pd
import fasttext
import sys # Used for progress logging
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# wget https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin -O lid.176.bin

# --- Configuration ---
MODEL_PATH = "lid.176.bin"  # path to downloaded fastText model
INPUT_FILE = "ALTO_page_stat_text.csv"
OUTPUT_FILE = "ALTO_page_lang_text_cs.csv"
CHUNK_SIZE = 1_000
THRESHOLD = 0.5  # confidence threshold for language detection
COMMON_LANGS = ["ces", "deu"]
COMMON_SUF = "Latn"
PERPLEXITY_THRESHOLD_MAX = 500  # lower to switch to the strict mode of lang suffixes
PERPLEXITY_THRESHOLD_MIN = 100  # lower to switch to the strict mode of is_text_good

# It's recommended to run this on a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "distilgpt2"
model_causal = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer_causal = AutoTokenizer.from_pretrained(model_id)


def calculate_perplexity(text, model=model_causal, tokenizer=tokenizer_causal):
    """Calculates the perplexity of a given text."""
    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    # The maximum sequence length the model can handle
    max_length = model.config.max_position_embeddings
    stride = 512  # How many tokens to slide the window by
    seq_len = input_ids.size(1)

    nlls = []  # Negative log likelihoods
    prev_end_loc = 0

    # Use a sliding window to process long texts
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100  # Mask tokens that were already calculated

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            # Loss is the average negative log likelihood
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).mean())
    result = ppl.item()

    print(result, text[:80])

    return result


# --- Load Model ---
print("Loading fastText model...")
try:
    model = fasttext.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except ValueError as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file 'lid.176.bin' is in the correct directory.")
    sys.exit(1) # Exit if the model can't be loaded

  
    
def detect_language(text):
    """Detect language and quality of OCR text with heuristics."""
    # Empty or NaN check
    if pd.isna(text) or not str(text).strip():
        return "", False

    clean_text = str(text).replace('\n', ' ').replace('\r', '').strip()
    n_chars = len(clean_text)

    # --- Pre-check: letter/digit/symbol ratio ---
    letters = sum(c.isalpha() for c in clean_text)
    digits = sum(c.isdigit() for c in clean_text)
    symbols = sum(not c.isalnum() and not c.isspace() for c in clean_text)
    spaces = sum(c.isspace() for c in clean_text)

    symbol_ratio = symbols / n_chars
    space_ratio = spaces / n_chars
    letter_ratio = letters / n_chars
    digit_ratio = digits / n_chars
    upper_ratio = sum(c.isupper() for c in clean_text) / letters if letters > 0 else 0.0

    if letter_ratio < 0.3 or digit_ratio > 0.4 or space_ratio > 0.3 or symbol_ratio > 0.2:
        return "NOISY_trash", False

    text_perplexity = calculate_perplexity(clean_text, model_causal, tokenizer_causal)

    # --- Language prediction ---
    labels, scores = model.predict(clean_text, k=2)
    lang_code = labels[0].replace("__label__", "")
    score1 = float(scores[0])
    score2 = float(scores[1]) if len(scores) > 1 else 0.0

    is_Latin = lang_code.split("_")[-1] == "Latn"
    is_common = lang_code.split("_")[0] in COMMON_LANGS

    base_lang = lang_code.split("_")[0].upper()
    if text_perplexity >= PERPLEXITY_THRESHOLD_MAX or upper_ratio > 0.9 or not is_Latin:
        base_lang += "_trash"
        return base_lang, False  # Gibberish detected
    elif text_perplexity >= PERPLEXITY_THRESHOLD_MIN or upper_ratio > 0.6 or not is_common:
        base_lang += "_noise"
        # continue to other checks

    # Confidence gap check
    if score1 - score2 < 0.15:
        base_lang += "_maybe"

    return base_lang, score1 >= THRESHOLD
    

# --- Main Processing Logic ---
try:
    # 1. Count total rows for progress tracking without loading the whole file
    print(f"Analyzing '{INPUT_FILE}'...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1 # Subtract 1 for the header row
    print(f"Found {total_rows:,} rows to process.")

    rows_processed = 0
    
    # 2. Process the file in chunks
    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(reader):
        if "text" not in chunk.columns:
            raise ValueError("Input CSV must have a 'text' column.")
        if "file" not in chunk.columns or "page" not in chunk.columns:
            raise ValueError("Input CSV must have 'file' and 'page' columns.")

        # Apply the function to the 'text' column
        predictions = chunk['text'].apply(detect_language)

        # Create the new columns from the prediction results
        chunk['lang'] = [p[0] for p in predictions]
        chunk['is_text_good'] = [p[1] for p in predictions]

        # 3. Save results for the current chunk immediately
        output_chunk = chunk[["file", "page", "lang", "is_text_good", "text"]]
        
        if i == 0:
            # For the first chunk, write with header
            output_chunk.to_csv(OUTPUT_FILE, index=False, mode='w', header=True)
        else:
            # For all subsequent chunks, append without header
            output_chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

        # 4. Update and log progress to the command line
        rows_processed += len(chunk)
        percentage = (rows_processed / total_rows) * 100 if total_rows > 0 else 100
        # Use sys.stdout.write and \r to print on the same line
        sys.stdout.write(f"\rProgress: {rows_processed:,}/{total_rows:,} rows ({percentage:.1f}%) processed.")
        sys.stdout.flush()

    # Print a final newline character to move off the progress line
    print("\nProcessing complete.")
    print(f"Results saved to {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"Error: Input file not found at '{INPUT_FILE}'")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
