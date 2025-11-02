#!/usr/bin/env bash
#
# text_api_calls.sh
#
# Purpose:
#   This script reads a CSV file, extracts text from a specified column,
#   and sends this text to two LINDAT/CLARIAH-CZ web APIs for linguistic
#   processing:
#
#   1. UDPipe: Performs tokenization, morphological analysis, and dependency
#              parsing. (Input: raw text, Output: CoNLL-U format)
#   2. NameTag: Performs Named Entity Recognition (NER) on the CoNLL-U output
#               from UDPipe. (Input: CoNLL-U, Output: conllu-ne format)
#
#   It is designed to be robust, handling large files, API rate limits,
#   and potential network errors.
#
# Features:
#   - Reads from a file or standard input (pipe).
#   - Automatically chunks large text blocks to fit API limits.
#   - Calls APIs with configurable retries and exponential backoff.
#   - Enforces a rate limit to avoid overloading the servers.
#   - Provides detailed progress monitoring and time estimation.
#   - Uses inline Python scripts for safe CSV reading and JSON parsing.
#
# Dependencies:
#   - curl: For making HTTP API requests.
#   - python3: For CSV/JSON parsing and helper logic.
#   - mktemp: For creating secure temporary files/directories.
#   - bc: (Optional, for better time-remaining estimates)
#
# Usage:
#   ./text_api_calls.sh input.csv [OUTDIR_NAMETAG] [OUTDIR_UDPIPE]
#   # Or, to process starting from a specific line (e.g., 100):
#   tail -n +100 input.csv | ./text_api_calls.sh - [OUTDIR_NAMETAG] [OUTDIR_UDPIPE]
#

# Stop the script immediately if any command fails (-e),
# or if an unset variable is used (-u).
# -o pipefail: ensures that a pipeline fails if *any* command in it fails.
set -euo pipefail

# ========== CONFIGURATION ==========
# These variables control the script's behavior.

# --- API Endpoints ---
UDPIPE_URL="https://lindat.mff.cuni.cz/services/udpipe/api/process"
NAMETAG_URL="https://lindat.mff.cuni.cz/services/nametag/api/recognize"

# --- Default Models ---
# These are used if the language-specific logic doesn't find a match.
DEFAULT_UDPIPE_MODEL="czech-pdt-ud-2.15-241121"
DEFAULT_UDPIPE_PARSER="czech-pdt-ud-2.15-241121"
DEFAULT_UDPIPE_TAGGER="czech-pdt-ud-2.15-241121"
DEFAULT_NAMETAG_MODEL="nametag3-multilingual-onto-250203"

# --- Output Formats ---
# This defines the processing pipeline:
# UDPipe takes "horizontal" (raw) text and outputs "conllu".
# NameTag takes "conllu" and outputs "conllu-ne".
NAMETAG_OUTPUT_FILE_FORMAT="conllu" # (This is a bit confusingly named)
UDPIPE_OUTPUT_FILE_FORMAT="conllu"
NAMETAG_OUTPUT_FORMAT="conllu-ne" # This is the *actual* format requested

# --- HTTP Settings ---
TIMEOUT=60            # Max seconds to wait for one API call
MAX_RETRIES=5         # Max times to retry a failed API call
BACKOFF_FACTOR=1.0    # How much to increase wait time after each failure
RATE_LIMIT_PER_SEC=5.0 # Max API calls per second (e.g., 5.0 = 1 call every 0.2s)
WORD_CHUNK_LIMIT=990  # Max words to send to UDPipe in one chunk

# --- CSV Settings ---
TEXT_COLUMN="text"    # Column name in the CSV that has the text
FILE_COLUMN="file"    # Column name for the file ID
PAGE_COLUMN="page"    # Column name for the page ID
LANGUAGE_COLUMN="lang" # Column name for the language code
MAX_ROWS_TO_PROCESS=450000 # Safety limit

# --- Processing Options ---
CHUNK_TEXT=true
MAX_TEXT_LENGTH=10000 # Truncate text longer than this
SKIP_EMPTY_TEXT=true  # Don't process rows with no text
# ========== END CONFIGURATION ==========


# ========== PROGRESS MONITORING FUNCTIONS ==========
# These functions print status updates to the console.

get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))

    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}
# Estimates time remaining based on average speed.
estimate_time_remaining() {
    local processed=$1
    local total=$2
    local elapsed=$3

    if [ $processed -le 0 ]; then echo "calculating..."; return; fi

    # Use 'bc' (basic calculator) for floating point math
    local rate=$(echo "scale=2; $processed / $elapsed" | bc -l 2>/dev/null || echo "0")
    local remaining=$((total - processed))

    if [ "$(echo "$rate > 0" | bc -l 2>/dev/null)" = "1" ]; then
        local eta=$(echo "scale=0; $remaining / $rate" | bc -l 2>/dev/null || echo "0")
        format_duration $eta
    else
        echo "calculating..."
    fi
}
# Prints a detailed progress block.
print_progress() {
    local processed=$1
    local total=$2
    local elapsed=$3
    local current_file=$4
    local current_page=$5

    local percentage=0
    if [ $total -gt 0 ]; then
        percentage=$(echo "scale=1; $processed * 100 / $total" | bc -l 2>/dev/null || echo "0")
    fi
    local rate=$(echo "scale=2; $processed / $elapsed" | bc -l 2>/dev/null || echo "0")
    local eta=$(estimate_time_remaining $processed $total $elapsed)

    printf '\n=== PROGRESS UPDATE [%s] ===\n' "$(get_timestamp)"
    printf 'Processed: %d / %d rows (%s%%)\n' "$processed" "$total" "$percentage"
    printf 'Current: %s (page %s)\n' "$current_file" "$current_page"
    printf 'Elapsed: %s\n' "$(format_duration $elapsed)"
    printf 'Rate: %s rows/sec\n' "$rate"
    printf 'ETA: %s\n' "$eta"
    printf '=====================================\n\n'
}
# Prints a simple one-line progress update.
print_simple_progress() {
    local processed=$1
    local elapsed=$2
    local current_file=$3
    local current_page=$4

    local rate=$(echo "scale=1; $processed / $elapsed" | bc -l 2>/dev/null || echo "0")
    printf '[%s] + + + Processed %d rows (%s/sec) + + + Current: %s:%s\n' \
           "$(get_timestamp)" "$processed" "$rate" "$current_file" "$current_page"
}
# Check if 'bc' is available. If not, use simpler progress functions.
if ! command -v bc >/dev/null 2>&1; then
    echo "Warning: 'bc' not found. Time estimates will be simplified." >&2
    # Redefine functions without 'bc'
    estimate_time_remaining() { echo "bc not available"; }
    print_progress() {
        # ... (simplified version) ...
    }
    print_simple_progress() {
        # ... (simplified version) ...
    }
fi
# ========== END PROGRESS FUNCTIONS ==========


# ========== ARGUMENT PARSING ==========
# Detect if input is coming from a pipe
stdin_is_pipe=false
if [ ! -t 0 ]; then
  stdin_is_pipe=true
fi
# Show usage if no args and no pipe
if [ "$#" -eq 0 ] && [ "$stdin_is_pipe" = false ]; then
  echo "Usage: $0 input.csv [OUTDIR_NAMETAG] [OUTDIR_UDPIPE]"
  echo "   or: tail -n +START input.csv | $0 - [OUTDIR_NAMETAG] [OUTDIR_UDPIPE]"
  exit 1
fi
# Handle input source (file vs. stdin)
if [ "$#" -ge 1 ] && [ "$1" = "-" ]; then
  INPUT_FROM_STDIN=true
  shift # Remove "-" from arguments
elif [ "$#" -ge 1 ]; then
  INPUT_FROM_STDIN=false
  INPUT_CSV="$1"
  shift # Remove filename from arguments
else
  INPUT_FROM_STDIN=true # Default to stdin if no args but pipe exists
fi

# Get output directory arguments (or use defaults)
OUTDIR_NT="${1:-../NameTag}"; shift || true
OUTDIR_UD="${1:-../UDPipe}"; shift || true
# Create output directories
mkdir -p "$OUTDIR_NT/onepagers" "$OUTDIR_UD/onepagers"
# ========== END ARGUMENT PARSING ==========


# ========== HELPER FUNCTIONS ==========

# Selects the correct UDPipe model based on language code
choose_udpipe_model() {
  local lang="$1"
  local lc=$(printf '%.3s' "$lang" | tr '[:upper:]' '[:lower:]') # get 3-letter lowercase
  case "$lc" in
    cs|ces) echo "czech-pdt-ud-2.15-241121" ;;
    en|eng) echo "english-ewt-ud-2.15-241121" ;;
    de|deu|ger) echo "german-gsd-ud-2.15-241121" ;;
    sk|slk) echo "slovak-snk-ud-2.15-241121" ;;
    *) echo "$DEFAULT_UDPIPE_MODEL" ;; # Fallback
  esac
}
# Selects the correct NameTag model based on language code
choose_nametag_model() {
  local lang="$1"
  local lc=$(printf '%.3s' "$lang" | tr '[:upper:]' '[:lower:]')
  case "$lc" in
    cs|ces) echo "nametag3-czech-cnec2.0-240830" ;;
    *) echo "$DEFAULT_NAMETAG_MODEL" ;; # Fallback
  esac
}
# Pauses the script to respect the API rate limit
rate_limit() {
  # Calculate delay needed (e.g., 1.0 / 5.0 = 0.2 seconds)
  local delay=$(python3 -c "print(1.0 / $RATE_LIMIT_PER_SEC)" 2>/dev/null || echo "0.2")
  sleep "$delay"
}

# --- This is a "Here Document" ---
# It creates a temporary Python script on the fly to read the CSV.
# This is *much* safer and more robust than trying to parse CSV with Bash.
PYFILE="$(mktemp --suffix=_csv_reader.py)"
trap 'rm -f "$PYFILE"' EXIT # Ensure the temp file is deleted when script exits
cat > "$PYFILE" <<'PY'
#!/usr/bin/env python3
import sys, csv, os
# This script will intelligently handle CSVs that are piped in
# without a header, by adding one if it's missing.
EXPECTED_HEADER = "file,page,lang,is_text_good,text\n"
def ensure_header(stream):
    first = stream.readline()
    if not first:
        return [EXPECTED_HEADER] # Handle empty input
    # Check if the first line looks like a header
    if "file" in first and "page" in first and "lang" in first and "text" in first:
        return [first] + list(stream) # Header is fine
    else:
        return [EXPECTED_HEADER, first] + list(stream) # Prepend header

def csv_reader_stream(f):
    # Use Python's built-in CSV reader
    rdr = csv.DictReader(f)
    row_count = 0
    for row in rdr:
        row_count += 1
        # Respect the safety limit from the shell environment
        if row_count > int(os.environ.get('MAX_ROWS_TO_PROCESS', '450000')):
            print(f"[INFO] Reached max rows limit: {row_count-1}", file=sys.stderr)
            break

        # Get data from columns, using environment variables for column names
        fn = (row.get(os.environ.get('FILE_COLUMN', 'file')) or '').strip()
        pg = (row.get(os.environ.get('PAGE_COLUMN', 'page')) or '').strip()
        lg = (row.get(os.environ.get('LANGUAGE_COLUMN', 'lang')) or '').strip()
        tx = row.get(os.environ.get('TEXT_COLUMN', 'text')) or ''

        # Clean the text: remove newlines
        tx = tx.replace('\r','').replace('\n',' ')

        # Skip row if text is empty
        if os.environ.get('SKIP_EMPTY_TEXT', 'true').lower() == 'true' and not tx.strip():
            continue

        # Truncate text if it's too long
        max_len = int(os.environ.get('MAX_TEXT_LENGTH', '10000'))
        if len(tx) > max_len:
            tx = tx[:max_len]

        # Print the data as simple Tab-Separated Values (TSV)
        # This is easy for the Bash 'read' command to parse.
        try:
            sys.stdout.write(f"{fn}\t{pg}\t{lg}\t{tx}\n")
        except BrokenPipeError:
            sys.exit(0) # Handle broken pipes gracefully

def main():
    if len(sys.argv) >= 2 and sys.argv[1] != '-':
        # Input is a file
        with open(sys.argv[1], encoding='utf-8', newline='') as f:
            csv_reader_stream(f)
    else:
        # Input is stdin (a pipe)
        lines = ensure_header(sys.stdin)
        csv_reader_stream(lines)

if __name__ == '__main__':
    main()
PY
chmod +x "$PYFILE" # Make the temporary Python script executable

# This function uses Python to safely parse the JSON response from the API.
parse_api_response() {
    local response_file="$1"
    local api_name="$2"
    if [ ! -s "$response_file" ]; then
        echo "[ERR] $api_name: Empty response" >&2
        return 1
    fi
    # Quick check: does it look like JSON?
    if ! head -c1 "$response_file" | grep -q '[{[]'; then
        echo "[ERR] $api_name: Response doesn't look like JSON:" >&2
        head -n 3 "$response_file" >&2
        return 1
    fi

    # Use Python to parse JSON and extract the 'result' field
    python3 -c "
import sys, json
try:
    with open('$response_file', 'r', encoding='utf-8') as f:
        j = json.load(f)
        result = j.get('result', '')
        if not result or not result.strip():
            # treat empty result as normal (skip silently)
            sys.exit(2) # Special exit code for "empty but OK"
        sys.stdout.write(result)
except json.JSONDecodeError as e:
    print(f'[ERR] $api_name: JSON decode error: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'[ERR] $api_name: Unexpected error: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# This is the robust API call function with retries.
api_call_with_retry() {
    local api_name="$1"
    local url="$2"
    local response_file="$3"
    shift 3  # Remove first 3 args, rest are curl parameters

    local attempt=1
    local delay=1

    while [ $attempt -le $MAX_RETRIES ]; do
        printf '[ATTEMPT %d/%d] |--==[ %s ]==--| call\n' "$attempt" "$MAX_RETRIES" "$api_name"

        # Use a separate file to store the HTTP status code
        local http_code_file="$response_file.httpcode"

        # Run curl with a timeout
        if timeout "$TIMEOUT" curl -s -S -w "%{http_code}" "$@" "$url" -o "$response_file" > "$http_code_file"; then
            local http_code=$(cat "$http_code_file")
            rm -f "$http_code_file"

            if [ "$http_code" = "200" ]; then
                return 0  # Success!
            else
                printf '[ERR] %s HTTP %s (attempt %d/%d)\n' "$api_name" "$http_code" "$attempt" "$MAX_RETRIES"
                if [ $attempt -eq $MAX_RETRIES ]; then
                    printf '[ERR] Response content:\n' >&2
                    cat "$response_file" >&2
                    return 1 # Final failure
                fi
            fi
        else
            printf '[ERR] %s curl failed (timeout or other error) (attempt %d/%d)\n' "$api_name" "$attempt" "$MAX_RETRIES"
            rm -f "$http_code_file"
            if [ $attempt -eq $MAX_RETRIES ]; then
                return 1 # Final failure
            fi
        fi

        # Wait before retrying (exponential backoff)
        sleep "$delay"
        delay=$(python3 -c "print(int($delay * $BACKOFF_FACTOR + 1))" 2>/dev/null || echo $((delay + 1)))
        attempt=$((attempt + 1))
    done

    return 1 # Should not be reached, but good practice
}
# ========== END HELPER FUNCTIONS ==========


# ========== MAIN PROCESSING ==========

# Export shell variables so the Python script can read them
export MAX_ROWS_TO_PROCESS FILE_COLUMN PAGE_COLUMN LANGUAGE_COLUMN TEXT_COLUMN
export SKIP_EMPTY_TEXT MAX_TEXT_LENGTH

# Define the command to *start* the process (either from file or stdin)
if [ "$INPUT_FROM_STDIN" = true ]; then
  exec_cmd=(python3 "$PYFILE" -)
else
  exec_cmd=(python3 "$PYFILE" "$INPUT_CSV")
fi

# Initialize progress tracking
processed_count=0
start_time=$(date +%s)
total_estimated=$MAX_ROWS_TO_PROCESS # Use this for progress %

printf '\n🚀 STARTING PROCESSING [%s] 🚀\n' "$(get_timestamp)"
printf 'Configuration:\n'
printf '  - Max rows: %d\n' "$MAX_ROWS_TO_PROCESS"
printf '  - Rate limit: %s calls/sec\n' "$RATE_LIMIT_PER_SEC"
printf '  - Output dirs: NT=%s, UD=%s\n' "$OUTDIR_NT" "$OUTDIR_UD"
printf '  - Pipeline: text → UDPipe (CoNLL-U) → NameTag (conllu-ne)\n'
printf '========================================\n\n'

# --- MAIN LOOP ---
# Run the Python CSV reader and pipe its TSV output to the 'while' loop
"${exec_cmd[@]}" | while IFS=$'\t' read -r file page lang text; do
  # Clean up any stray carriage returns
  file=$(printf '%s' "$file" | tr -d '\r\n')
  page=$(printf '%s' "$page" | tr -d '\r\n')
  lang=$(printf '%s' "$lang" | tr -d '\r\n')

  processed_count=$((processed_count + 1))
  current_time=$(date +%s)
  elapsed=$((current_time - start_time))

  # --- Progress Reporting ---
  if [ $((processed_count % PROGRESS_INTERVAL)) -eq 0 ]; then
    if [ $((processed_count % DETAILED_PROGRESS_INTERVAL)) -eq 0 ]; then
      print_progress $processed_count $total_estimated $elapsed "$file" "$page"
    else
      print_simple_progress $processed_count $elapsed "$file" "$page"
    fi
  fi

  # --- Output Path Logic ---
  # Page 1 goes into a 'onepagers' flat directory
  if [ "$page" = "1" ] || [ "$page" = "01" ]; then
    nt_dir="$OUTDIR_NT/onepagers"
    ud_dir="$OUTDIR_UD/onepagers"
  else
    # Other pages go into subfolders named after the file ID
    nt_dir="$OUTDIR_NT/$file"
    ud_dir="$OUTDIR_UD/$file"
    mkdir -p "$nt_dir" "$ud_dir"
  fi
  ud_out="$ud_dir/$file-$page.$UDPIPE_OUTPUT_FILE_FORMAT"
  nt_out="$nt_dir/$file-$page.$NAMETAG_OUTPUT_FILE_FORMAT"

  if [ -z "$text" ] && [ "$SKIP_EMPTY_TEXT" = "true" ]; then
    continue # Skip this row
  fi

  # Get the correct API models for this row's language
  UDPIPE_MODEL=$(choose_udpipe_model "$lang")
  NAMETAG_MODEL=$(choose_nametag_model "$lang")

  # Create a unique temporary directory for this *single row*
  tmpd=$(mktemp -d) || { printf '[ERR] [%s] mktemp failed for %s page=%s\n' "$(get_timestamp)" "$file" "$page"; continue; }

  # Define a cleanup function for this row
  cleanup_tmpd() {
    if [ -d "$tmpd" ]; then
      rm -rf "$tmpd"
    fi
  }

  # Save this row's text to a temporary file
  tmpf="$tmpd/input.txt"
  printf '%s' "$text" > "$tmpf"

  if [ ! -s "$tmpf" ]; then
    printf '[ERR] [%s] text empty for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
    cleanup_tmpd
    continue
  fi

  # --- STEP 1: UDPipe (text → CoNLL-U) ---
  if [ ! -f "$ud_out" ]; then # Only run if output doesn't exist
    printf '[UDPIPE] model=%s file=%s page=%s\n' "$UDPIPE_MODEL" "$file" "$page"

    # Use an inline Python script to chunk the text file by word count
    if ! python3 - "$tmpf" "$tmpd" "$WORD_CHUNK_LIMIT" <<'PY'
import sys, os, re
try:
    infile = sys.argv[1]
    outdir = sys.argv[2]
    maxw = int(sys.argv[3])
    with open(infile, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text: sys.exit(1)
    words = re.split(r'\s+', text) # Split by any whitespace
    if not words: sys.exit(1)
    os.makedirs(outdir, exist_ok=True)
    chunk_count = 0
    for i in range(0, len(words), maxw):
        chunk_text = " ".join(words[i:i+maxw]).strip()
        if not chunk_text: continue
        with open(os.path.join(outdir, f"chunk{chunk_count}.txt"), "w", encoding="utf-8") as of:
            of.write(chunk_text)
        chunk_count += 1
    if chunk_count == 0: sys.exit(2)
except Exception as e:
    sys.stderr.write(f"[ERR] chunker failed: {e}\\n")
    sys.exit(3)
PY
    then
      printf '[ERR] [%s] failed to split text for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      cleanup_tmpd
      continue
    fi

    # Check if chunks were actually created
    if ! ls "$tmpd"/chunk*.txt >/dev/null 2>&1; then
      printf '[ERR] [%s] no chunks produced for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      cleanup_tmpd
      continue
    fi

    rm -f "$ud_out" # Remove partial file if it exists
    success=true

    # Process each chunk and append results to the final CoNLL-U file
    for cf in "$tmpd"/chunk*.txt; do
      [ -s "$cf" ] || continue # Skip empty chunks
      response_file="$tmpd/udpipe_response_$(basename "$cf").json"

      # Call UDPipe API
      if api_call_with_retry "UDPipe" "$UDPIPE_URL" "$response_file" \
         -F "data=@${cf}" -F "input=horizontal" -F "output=conllu" \
         -F "model=${UDPIPE_MODEL}" -F "parser=" -F "tagger=" -F "tokenizer="; then

        # Parse the JSON response and append the 'result' to our output file
        if parse_api_response "$response_file" "UDPipe" >> "$ud_out"; then
          continue # Success, move to next chunk
        elif [ $? -eq 2 ]; then
          # Empty result (code 2), skip silently
          printf '[SKIP] [%s] UDPipe returned empty result for chunk: %s\n' "$(get_timestamp)" "$(basename "$cf")"
        else
          printf '[ERR] [%s] UDPipe parsing failed for chunk: %s\n' "$(get_timestamp)" "$(basename "$cf")"
          success=false
          break # Stop processing chunks for this file
        fi
      else
        printf '[ERR] [%s] UDPipe API call failed for %s page=%s (chunk=%s)\n' "$(get_timestamp)" "$file" "$page" "$(basename "$cf")"
        success=false
        break # Stop processing chunks for this file
      fi

      rate_limit # Pause between chunks
    done

    # Check if the overall process for this row succeeded
    if [ "$success" = true ] && [ -f "$ud_out" ] && [ -s "$ud_out" ]; then
      printf '[WRITE] [%s] UD CoNLL-U output: %s\n' "$(get_timestamp)" "$ud_out"
    else
      printf '[ERR] [%s] UDPipe processing failed for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      rm -f "$ud_out" # Delete partial/failed file
      cleanup_tmpd
      continue # Skip to the next row in the CSV
    fi
  fi

  # --- STEP 2: NameTag (CoNLL-U → conllu-ne) ---
  if [ ! -f "$nt_out" ]; then # Only run if output doesn't exist
    if [ ! -f "$ud_out" ] || [ ! -s "$ud_out" ]; then
      printf '[ERR] [%s] Cannot run NameTag: UDPipe output missing for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      cleanup_tmpd
      continue # Skip to the next row
    fi

    printf '[NAMETAG] model=%s file=%s page=%s (processing CoNLL-U)\n' "$NAMETAG_MODEL" "$file" "$page"
    response_file="$tmpd/nametag_response.json"

    # Call NameTag API, sending the *entire* CoNLL-U file from UDPipe
    if api_call_with_retry "NameTag" "$NAMETAG_URL" "$response_file" \
       -F "data=@${ud_out}" -F "input=conllu" -F "model=${NAMETAG_MODEL}" -F "output=${NAMETAG_OUTPUT_FORMAT}"; then

      # Parse the response and write it to the final output file
      if parse_api_response "$response_file" "NameTag" > "$nt_out"; then
        printf '[WRITE] [%s] NT conllu-ne output: %s\n' "$(get_timestamp)" "$nt_out"
      elif [ $? -eq 2 ]; then
        # Empty result, don't save the file
        printf '[SKIP] [%s] NameTag returned empty result for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
        rm -f "$nt_out"
      else
        printf '[ERR] [%s] NameTag parsing failed for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
        rm -f "$nt_out" # Delete failed file
      fi
    else
      printf '[ERR] [%s] NameTag API call failed for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      rm -f "$nt_out"
    fi
    rate_limit # Pause after this row
  fi

  # Clean up the temporary directory for this row
  cleanup_tmpd
done || true # '|| true' prevents the script from exiting if the 'read' command fails (e.g., on a bad pipe)

# --- FINAL SUMMARY ---
end_time=$(date +%s)
total_elapsed=$((end_time - start_time))
printf '\n🏁 PROCESSING COMPLETE [%s] 🏁\n' "$(get_timestamp)"
printf 'Total processed: %d rows\n' "$processed_count"
printf 'Total time: %s\n' "$(format_duration $total_elapsed)"
if [ $total_elapsed -gt 0 ]; then
    final_rate=$(echo "scale=2; $processed_count / $total_elapsed" | bc -l 2>/dev/null || echo "N/A")
    printf 'Average rate: %s rows/sec\n' "$final_rate"
fi
printf 'NT outputs under: %s\n' "$OUTDIR_NT"
printf 'UD outputs under: %s\n' "$OUTDIR_UD"
printf '=========================================\n'