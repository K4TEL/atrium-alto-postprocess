#!/bin/bash
# CSV Text Merger Shell Script with ALTO processing
# Usage: ./addtext.sh input.csv [output.csv]

INPUT_CSV="$1"
OUTPUT_CSV="${2:-${INPUT_CSV%.*}_with_text.csv}"

# Check if input file is provided
if [ -z "$INPUT_CSV" ]; then
    echo "Usage: $0 input.csv [output.csv]"
    echo "Example: $0 data.csv data_with_text.csv"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input file '$INPUT_CSV' not found!"
    exit 1
fi

# Check if alto-tools is available
if ! command -v alto-tools &> /dev/null; then
    echo "Error: alto-tools command not found!"
    echo "Please install alto-tools or make sure it's in your PATH."
    exit 1
fi

echo "Input CSV: $INPUT_CSV"
echo "Output CSV: $OUTPUT_CSV"
echo "Starting processing..."

# Create temporary file for processing
TEMP_FILE=$(mktemp ./temp_alto_table_XXXXXX.csv)
TEMP_UNSORTED=$(mktemp ./temp_unsorted_alto_table_XXXXXX.csv)

# Add header without path column, with text column at the end
head -n 1 "$INPUT_CSV" | sed 's/,path$/,text/' > "$TEMP_UNSORTED"

# Process each line (skip header)
line_count=0
processed_count=0
missing_count=0
alto_error_count=0

tail -n +2 "$INPUT_CSV" | while IFS=',' read -r file page textlines illustrations graphics strings path; do
    ((line_count++))
    
    text_content=""
    
    # Check if path exists and process with alto-tools
    alto_file_found=""
    if [ -f "$path" ]; then
        alto_file_found="$path"
    else
        # Try alternative path in onepagers directory
        filename=$(basename "$path")
        alternative_path="../ALTO/A-PAGE/onepagers/$filename"
        if [ -f "$alternative_path" ]; then
            alto_file_found="$alternative_path"
        fi
    fi
    
    if [ -n "$alto_file_found" ]; then
        # Use alto-tools to extract text from ALTO XML
        alto_text=$(alto-tools -t "$alto_file_found" 2>/dev/null)
        alto_exit_code=$?
        
        if [ $alto_exit_code -eq 0 ]; then
            # alto-tools succeeded, process the text (even if empty)
            if [ -n "$alto_text" ]; then
                # Preprocess text for CSV:
                # 1. Replace newlines with spaces
                # 2. Remove carriage returns
                # 3. Escape double quotes by doubling them
                # 4. Remove any remaining commas that could break CSV structure
                text_content=$(echo "$alto_text" | \
                    tr '\n\r' '  ' | \
                    sed 's/"/""""/g' | \
                    tr ',' ' ')
            else
                # Empty text content - this is valid, just leave text_content empty
                text_content=""
            fi
            ((processed_count++))
        else
            echo "Warning: Failed to extract text from ALTO file: $alto_file_found" >&2
            ((alto_error_count++))
        fi
    else
        echo "Warning: ALTO file not found in either location: $path or ../ALTO/A-PAGE/onepagers/$(basename "$path")" >&2
        ((missing_count++))
    fi
    
    # Write row without path column, with text content at the end (properly quoted)
    echo "${file},${page},${textlines},${strings},${glyphs},${illustrations},${graphics},\"${text_content}\"" >> "$TEMP_UNSORTED"
    
    # Progress indicator
    if [ $((line_count % 100)) -eq 0 ]; then
        echo "Processed $line_count rows..." >&2
    fi
done

# Wait for background process to complete
wait

# Capture the counters from the subshell (since the while loop runs in a subshell, variables don't persist)
line_count=$(tail -n +2 "$INPUT_CSV" | wc -l)

echo ""
echo "Processing complete! Now sorting the CSV..."

# Sort the CSV file by first two columns (file and page) while preserving header
(head -1 "$TEMP_UNSORTED" && tail -n +2 "$TEMP_UNSORTED" | sort -t',' -k1,1 -k2,2n) > "$OUTPUT_CSV"

# Clean up temporary files
rm -f "$TEMP_FILE" "$TEMP_UNSORTED"

echo "Sorting complete!"
echo "Output written to: $OUTPUT_CSV"
echo "Total rows processed: $line_count"

# Note: Since the processing loop runs in a subshell, the individual counters 
# (processed_count, missing_count, alto_error_count) are not accessible here.
# If you need these statistics, consider using a different approach like 
# writing counts to temporary files or restructuring the loop.

echo ""
echo "Check above output for any warnings about ALTO file processing."
