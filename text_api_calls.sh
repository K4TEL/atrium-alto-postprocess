#!/usr/bin/env bash
# process_lindat_csv_simple.sh (with integrated configuration and progress monitoring)
#
# Usage:
#   ./text_api_calls.sh input.csv [OUTDIR_NAMETAG] [OUTDIR_UDPIPE]
#   tail -n +START input.csv | ./text_api_calls.sh - [OUTDIR_NAMETAG] [OUTDIR_UDPIPE]
#
# Now automatically injects the CSV header if missing when reading from stdin.

set -euo pipefail

# ========== CONFIGURATION (from api_config.txt) ==========

# API endpoints
UDPIPE_URL="https://lindat.mff.cuni.cz/services/udpipe/api/process"
NAMETAG_URL="https://lindat.mff.cuni.cz/services/nametag/api/recognize"

# Default models
DEFAULT_UDPIPE_MODEL="czech-pdt-ud-2.15-241121"
DEFAULT_UDPIPE_PARSER="czech-pdt-ud-2.15-241121"
DEFAULT_UDPIPE_TAGGER="czech-pdt-ud-2.15-241121"
DEFAULT_NAMETAG_MODEL="nametag3-multilingual-onto-250203"

# Common settings
COMMON_LANG="CES"
NAMETAG_OUTPUT_FILE_FORMAT="txt"
UDPIPE_OUTPUT_FILE_FORMAT="txt"
NAMETAG_OUTPUT_FORMAT="vertical"

# HTTP settings
TIMEOUT=60
MAX_RETRIES=5
BACKOFF_FACTOR=1.0
RATE_LIMIT_PER_SEC=5.0
WORD_CHUNK_LIMIT=990

# CSV settings
TEXT_COLUMN="text"
FILE_COLUMN="file"
PAGE_COLUMN="page"
LANGUAGE_COLUMN="lang"
MAX_ROWS_TO_PROCESS=450000

# Processing options
CHUNK_TEXT=true
MAX_TEXT_LENGTH=10000
SKIP_EMPTY_TEXT=true

# Progress monitoring settings
PROGRESS_INTERVAL=500  # Report progress every N rows
DETAILED_PROGRESS_INTERVAL=1000  # Detailed progress every N rows

# ========== END CONFIGURATION ==========

# Progress monitoring functions
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

estimate_time_remaining() {
    local processed=$1
    local total=$2
    local elapsed=$3
    
    if [ $processed -le 0 ]; then
        echo "calculating..."
        return
    fi
    
    local rate=$(echo "scale=2; $processed / $elapsed" | bc -l 2>/dev/null || echo "0")
    local remaining=$((total - processed))
    
    if [ "$(echo "$rate > 0" | bc -l 2>/dev/null)" = "1" ]; then
        local eta=$(echo "scale=0; $remaining / $rate" | bc -l 2>/dev/null || echo "0")
        format_duration $eta
    else
        echo "calculating..."
    fi
}

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

print_simple_progress() {
    local processed=$1
    local elapsed=$2
    local current_file=$3
    local current_page=$4
    
    local rate=$(echo "scale=1; $processed / $elapsed" | bc -l 2>/dev/null || echo "0")
    printf '[%s] + + + Processed %d rows (%s/sec) + + + Current: %s:%s\n' \
           "$(get_timestamp)" "$processed" "$rate" "$current_file" "$current_page"
}

# Check if bc is available for calculations
if ! command -v bc >/dev/null 2>&1; then
    echo "Warning: 'bc' not found. Time estimates will be simplified." >&2
    
    # Fallback functions without bc
    estimate_time_remaining() {
        echo "bc not available"
    }
    
    print_progress() {
        local processed=$1
        local total=$2
        local elapsed=$3
        local current_file=$4
        local current_page=$5
        
        printf '\n=== PROGRESS UPDATE [%s] ===\n' "$(get_timestamp)"
        printf 'Processed: %d / %d rows\n' "$processed" "$total"
        printf 'Current: %s (page %s)\n' "$current_file" "$current_page"
        printf 'Elapsed: %s\n' "$(format_duration $elapsed)"
        printf '=====================================\n\n'
    }
    
    print_simple_progress() {
        local processed=$1
        local elapsed=$2
        local current_file=$3
        local current_page=$4
        
        printf '[%s] Processed %d rows - Current: %s:%s\n' \
               "$(get_timestamp)" "$processed" "$current_file" "$current_page"
    }
fi

stdin_is_pipe=false
if [ ! -t 0 ]; then
  stdin_is_pipe=true
fi

if [ "$#" -eq 0 ] && [ "$stdin_is_pipe" = false ]; then
  echo "Usage: $0 input.csv [OUTDIR_NAMETAG] [OUTDIR_UDPIPE]"
  echo "   or: tail -n +START input.csv | $0 - [OUTDIR_NAMETAG] [OUTDIR_UDPIPE]"
  exit 1
fi

if [ "$#" -ge 1 ] && [ "$1" = "-" ]; then
  INPUT_FROM_STDIN=true
  shift
elif [ "$#" -ge 1 ]; then
  INPUT_FROM_STDIN=false
  INPUT_CSV="$1"
  shift
else
  INPUT_FROM_STDIN=true
fi

OUTDIR_NT="${1:-../NameTag}"; shift || true
OUTDIR_UD="${1:-../UDPipe}"; shift || true

mkdir -p "$OUTDIR_NT/onepagers" "$OUTDIR_UD/onepagers"

# Model selection based on language with fallback to defaults
choose_udpipe_model() {
  local lang="$1"
  local lc=$(printf '%.3s' "$lang" | tr '[:upper:]' '[:lower:]')
  case "$lc" in
    cs|ces) echo "czech-pdt-ud-2.15-241121" ;;
    en|eng) echo "english-ewt-ud-2.15-241121" ;;
    de|deu|ger) echo "german-gsd-ud-2.15-241121" ;;
    sk|slk) echo "slovak-snk-ud-2.15-241121" ;;
    *) echo "$DEFAULT_UDPIPE_MODEL" ;;
  esac
}

choose_nametag_model() {
  local lang="$1"
  local lc=$(printf '%.3s' "$lang" | tr '[:upper:]' '[:lower:]')
  case "$lc" in
    cs|ces) echo "nametag3-czech-cnec2.0-240830" ;;
    *) echo "$DEFAULT_NAMETAG_MODEL" ;;
  esac
}

# Rate limiting function
rate_limit() {
  local delay=$(python3 -c "print(1.0 / $RATE_LIMIT_PER_SEC)" 2>/dev/null || echo "0.2")
  sleep "$delay"
}

PYFILE="$(mktemp --suffix=_csv_reader.py)"
trap 'rm -f "$PYFILE"' EXIT

cat > "$PYFILE" <<'PY'
#!/usr/bin/env python3
import sys, csv, os

EXPECTED_HEADER = "file,page,lang,is_text_good,text\n"

def ensure_header(stream):
    first = stream.readline()
    if not first:
        return [EXPECTED_HEADER]
    if "file" in first and "page" in first and "lang" in first and "text" in first:
        return [first] + list(stream)
    else:
        return [EXPECTED_HEADER, first] + list(stream)

def csv_reader_stream(f):
    rdr = csv.DictReader(f)
    row_count = 0
    for row in rdr:
        row_count += 1
        if row_count > int(os.environ.get('MAX_ROWS_TO_PROCESS', '450000')):
            print(f"[INFO] Reached max rows limit: {row_count-1}", file=sys.stderr)
            break
        fn = (row.get(os.environ.get('FILE_COLUMN', 'file')) or '').strip()
        pg = (row.get(os.environ.get('PAGE_COLUMN', 'page')) or '').strip()
        lg = (row.get(os.environ.get('LANGUAGE_COLUMN', 'lang')) or '').strip()
        tx = row.get(os.environ.get('TEXT_COLUMN', 'text')) or ''
        tx = tx.replace('\r','').replace('\n',' ')
        if os.environ.get('SKIP_EMPTY_TEXT', 'true').lower() == 'true' and not tx.strip():
            continue
        max_len = int(os.environ.get('MAX_TEXT_LENGTH', '10000'))
        if len(tx) > max_len:
            tx = tx[:max_len]
        try:
            sys.stdout.write(f"{fn}\t{pg}\t{lg}\t{tx}\n")
        except BrokenPipeError:
            sys.exit(0)

def main():
    if len(sys.argv) >= 2 and sys.argv[1] != '-':
        with open(sys.argv[1], encoding='utf-8', newline='') as f:
            csv_reader_stream(f)
    else:
        lines = ensure_header(sys.stdin)
        csv_reader_stream(lines)

if __name__ == '__main__':
    main()
PY
chmod +x "$PYFILE"

# Function to parse JSON response with better error handling
parse_api_response() {
    local response_file="$1"
    local api_name="$2"
    if [ ! -s "$response_file" ]; then
        echo "[ERR] $api_name: Empty response" >&2
        return 1
    fi
    if ! head -c1 "$response_file" | grep -q '[{[]'; then
        echo "[ERR] $api_name: Response doesn't look like JSON:" >&2
        head -n 3 "$response_file" >&2
        return 1
    fi
    python3 -c "
import sys, json
try:
    with open('$response_file', 'r', encoding='utf-8') as f:
        j = json.load(f)
        result = j.get('result', '')
        if not result or not result.strip():
            # treat empty result as normal (skip silently)
            sys.exit(2)
        sys.stdout.write(result)
except json.JSONDecodeError as e:
    print(f'[ERR] $api_name: JSON decode error: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'[ERR] $api_name: Unexpected error: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Retry function with exponential backoff
api_call_with_retry() {
    local api_name="$1"
    local url="$2"
    local response_file="$3"
    shift 3  # Remove first 3 args, rest are curl parameters
    
    local attempt=1
    local delay=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        printf '[ATTEMPT %d/%d] |--==[ %s ]==--| call\n' "$attempt" "$MAX_RETRIES" "$api_name"
        
        # Use separate file for HTTP code to avoid mixing with response
        local http_code_file="$response_file.httpcode"
        
        if timeout "$TIMEOUT" curl -s -S -w "%{http_code}" "$@" "$url" -o "$response_file" > "$http_code_file"; then
            local http_code=$(cat "$http_code_file")
            rm -f "$http_code_file"
            
            if [ "$http_code" = "200" ]; then
                return 0  # Success
            else
                printf '[ERR] %s HTTP %s (attempt %d/%d)\n' "$api_name" "$http_code" "$attempt" "$MAX_RETRIES"
                if [ $attempt -eq $MAX_RETRIES ]; then
                    printf '[ERR] Response content:\n' >&2
                    cat "$response_file" >&2
                    return 1
                fi
            fi
        else
            printf '[ERR] %s curl failed (attempt %d/%d)\n' "$api_name" "$attempt" "$MAX_RETRIES"
            rm -f "$http_code_file"
            if [ $attempt -eq $MAX_RETRIES ]; then
                return 1
            fi
        fi
        
        # Exponential backoff
        sleep "$delay"
        delay=$(python3 -c "print(int($delay * $BACKOFF_FACTOR + 1))" 2>/dev/null || echo $((delay + 1)))
        attempt=$((attempt + 1))
    done
    
    return 1
}

# Export environment variables for Python script
export MAX_ROWS_TO_PROCESS FILE_COLUMN PAGE_COLUMN LANGUAGE_COLUMN TEXT_COLUMN
export SKIP_EMPTY_TEXT MAX_TEXT_LENGTH

if [ "$INPUT_FROM_STDIN" = true ]; then
  exec_cmd=(python3 "$PYFILE" -)
else
  exec_cmd=(python3 "$PYFILE" "$INPUT_CSV")
fi

# Initialize progress tracking
processed_count=0
start_time=$(date +%s)
total_estimated=$MAX_ROWS_TO_PROCESS

printf '\n🚀 STARTING PROCESSING [%s] 🚀\n' "$(get_timestamp)"
printf 'Configuration:\n'
printf '  - Max rows: %d\n' "$MAX_ROWS_TO_PROCESS"
printf '  - Rate limit: %s calls/sec\n' "$RATE_LIMIT_PER_SEC"
printf '  - Output dirs: NT=%s, UD=%s\n' "$OUTDIR_NT" "$OUTDIR_UD"
printf '  - Progress updates every %d rows\n' "$PROGRESS_INTERVAL"
printf '========================================\n\n'

"${exec_cmd[@]}" | while IFS=$'\t' read -r file page lang text; do
  file=$(printf '%s' "$file" | tr -d '\r\n')
  page=$(printf '%s' "$page" | tr -d '\r\n')
  lang=$(printf '%s' "$lang" | tr -d '\r\n')
  
  processed_count=$((processed_count + 1))
  current_time=$(date +%s)
  elapsed=$((current_time - start_time))
  
  # Progress reporting
  if [ $((processed_count % PROGRESS_INTERVAL)) -eq 0 ]; then
    if [ $((processed_count % DETAILED_PROGRESS_INTERVAL)) -eq 0 ]; then
      print_progress $processed_count $total_estimated $elapsed "$file" "$page"
    else
      print_simple_progress $processed_count $elapsed "$file" "$page"
    fi
  fi

  if [ "$page" = "1" ] || [ "$page" = "01" ]; then
    nt_dir="$OUTDIR_NT/onepagers"
    ud_dir="$OUTDIR_UD/onepagers"
  else
    nt_dir="$OUTDIR_NT/$file"
    ud_dir="$OUTDIR_UD/$file"
    mkdir -p "$nt_dir" "$ud_dir"
  fi

  ud_out="$ud_dir/$file-$page.$UDPIPE_OUTPUT_FILE_FORMAT"
  nt_out="$nt_dir/$file-$page.$NAMETAG_OUTPUT_FILE_FORMAT"

  if [ -z "$text" ] && [ "$SKIP_EMPTY_TEXT" = "true" ]; then
    # printf '[SKIP] [%s] empty text for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
    continue
  fi

  UDPIPE_MODEL=$(choose_udpipe_model "$lang")
  NAMETAG_MODEL=$(choose_nametag_model "$lang")

  # Create tmp dir per row (unique and safe)
  tmpd=$(mktemp -d) || { printf '[ERR] [%s] mktemp failed for %s page=%s\n' "$(get_timestamp)" "$file" "$page"; continue; }
  
  # Cleanup function for this row
  cleanup_tmpd() { 
    if [ -d "$tmpd" ]; then
      rm -rf "$tmpd"
    fi
  }

  # Save input text to a stable file for all tools to use
  tmpf="$tmpd/input.txt"
  printf '%s' "$text" > "$tmpf"

  # Basic sanity checks
  if [ ! -s "$tmpf" ]; then
    printf '[ERR] [%s] text empty for %s page=%s (length=%s)\n' "$(get_timestamp)" "$file" "$page" "$(wc -c < "$tmpf" 2>/dev/null || echo 0)"
    cleanup_tmpd
    continue
  fi

  # --- UDPipe ---
  if [ ! -f "$ud_out" ]; then
    printf '[UDPIPE] model=%s file=%s page=%s\n' "$UDPIPE_MODEL" "$file" "$page"

    # Chunk input by reading the stable tmpf (avoid piping issues)
    if ! python3 - "$tmpf" "$tmpd" "$WORD_CHUNK_LIMIT" <<'PY'
import sys, os, re
try:
    infile = sys.argv[1]
    outdir = sys.argv[2]
    maxw = int(sys.argv[3])
    with open(infile, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        sys.stderr.write("[ERR] no input text provided\n")
        sys.exit(1)
    # Split on any whitespace
    words = re.split(r'\s+', text)
    if not words:
        sys.stderr.write("[ERR] no words after splitting\n")
        sys.exit(1)
    os.makedirs(outdir, exist_ok=True)
    chunk_count = 0
    for i in range(0, len(words), maxw):
        chunk_words = words[i:i+maxw]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words).strip()
        if not chunk_text:
            continue
        with open(os.path.join(outdir, f"chunk{chunk_count}.txt"), "w", encoding="utf-8") as of:
            of.write(chunk_text)
        chunk_count += 1
    # If nothing produced for some reason, exit non-zero
    if chunk_count == 0:
        sys.stderr.write("[ERR] no chunks produced by chunker\n")
        sys.exit(2)
except Exception as e:
    sys.stderr.write(f"[ERR] chunker failed: {e}\\n")
    sys.exit(3)
PY
    then
      printf '[ERR] [%s] failed to split text for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      cleanup_tmpd
      continue
    fi

    # Confirm chunks exist
    if ! ls "$tmpd"/chunk*.txt >/dev/null 2>&1; then
      printf '[ERR] [%s] no chunks produced for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      cleanup_tmpd
      continue
    fi

    rm -f "$ud_out"
    success=true

    # Process each chunk and accumulate results
    for cf in "$tmpd"/chunk*.txt; do
      [ -s "$cf" ] || continue
      response_file="$tmpd/udpipe_response_$(basename "$cf").json"
      
      if api_call_with_retry "UDPipe" "$UDPIPE_URL" "$response_file" \
         -F "data=@${cf}" -F "input=horizontal" -F "model=${UDPIPE_MODEL}" -F "parser=" -F "tagger=" -F "tokenizer="; then
        
        # Parse and append result
        if parse_api_response "$response_file" "UDPipe" >> "$ud_out"; then
          continue
          # printf '[CHUNK] [%s] UDPipe processed chunk: %s\n' "$(get_timestamp)" "$(basename "$cf")"
        elif [ $? -eq 2 ]; then
          # Empty result - skip silently
          printf '[SKIP] [%s] UDPipe returned empty result for chunk: %s\n' "$(get_timestamp)" "$(basename "$cf")"
        else
          printf '[ERR] [%s] UDPipe parsing failed for chunk: %s\n' "$(get_timestamp)" "$(basename "$cf")"
          success=false
          break
        fi
      else
        printf '[ERR] [%s] UDPipe API call failed for %s page=%s (chunk=%s)\n' "$(get_timestamp)" "$file" "$page" "$(basename "$cf")"
        success=false
        break
      fi
      
      rate_limit
    done

    if [ "$success" = true ] && [ -f "$ud_out" ] && [ -s "$ud_out" ]; then
      printf '[WRITE] [%s] UD output: %s\n' "$(get_timestamp)" "$ud_out"
    else
      printf '[ERR] [%s] UDPipe processing failed for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      rm -f "$ud_out"
    fi

  else
    # printf '[SKIP] [%s] UD output exists: %s\n' "$(get_timestamp)" "$ud_out"
    continue
  fi

  # --- NameTag ---
  # Reuse tmpf (the stable input file) so NameTag sees the exact same text
  if [ ! -f "$nt_out" ]; then
    printf '[NAMETAG] model=%s file=%s page=%s\n' "$NAMETAG_MODEL" "$file" "$page"
    response_file="$tmpd/nametag_response.json"

    if api_call_with_retry "NameTag" "$NAMETAG_URL" "$response_file" \
       -F "data=@${tmpf}" -F "model=${NAMETAG_MODEL}" -F "output=${NAMETAG_OUTPUT_FORMAT}"; then

      if parse_api_response "$response_file" "NameTag" > "$nt_out"; then
        printf '[WRITE] [%s] NT output: %s\n' "$(get_timestamp)" "$nt_out"
      elif [ $? -eq 2 ]; then
        # Empty result - don't save file
        printf '[SKIP] [%s] NameTag returned empty result for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
        rm -f "$nt_out"
      else
        printf '[ERR] [%s] NameTag parsing failed for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
        rm -f "$nt_out"
      fi
    else
      printf '[ERR] [%s] NameTag API call failed for %s page=%s\n' "$(get_timestamp)" "$file" "$page"
      rm -f "$nt_out"
    fi

    rate_limit
  else
    # printf '[SKIP] [%s] NT output exists: %s\n' "$(get_timestamp)" "$nt_out"
    continue
  fi

  # cleanup tmp dir for this row
  cleanup_tmpd

done || true

# Final summary
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
