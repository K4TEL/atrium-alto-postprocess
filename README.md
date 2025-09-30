# ALTO XML files postprocessing

## Setup

In a new virtual environment, while in the project directory, run:

    pip install -r requirements.txt

Then: 

    git clone https://github.com/cneud/alto-tools.git
    cd alto-tools
    pip install .
    
Additionally, somewhere in the project directory: 

    wget https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin -O lid.176.bin
    
You are good to go. 

## Split document-specific ALTOs into page-specific XML files

Ensure you have a directory of `<file>.alto.xml` files and then run:

    python3 page_split.py <input_dir> <output_dir>

Which will create: 

    <output_dir>
    ├── <file1>
        ├── <file1>-<page>.alto.xml 
        └── ...
    ├── <file2>
        ├── <file2>-<page>.alto.xml 
        └── ...
    └── ...

Where each of the page-specific files starts with a header taken from the source document-specific ALTO XML. 

## Get the statistics table of pages

Now the previous output directory is the input directory:

    python3 alto_stats_create.py <input_dir> -o output.csv

Which will start writing a CSV file line-by-line:

    file, page, textlines, illustrations, graphics, strings, path
    CTX200205348, 1, 33, 1, 10, 163, /lnet/.../A-PAGE/CTX200205348/CTX200205348-1.alto.xml
    CTX200205348, 2, 0, 1, 12, 0, /lnet/.../A-PAGE/CTX200205348/CTX200205348-2.alto.xml
    CTX200603119, 3, 29, 2, 30, 206, /lnet/.../A-PAGE/CTX200603119/CTX200603119-3.alto.xml

The framework used for statistics and text contents extraction from XML files 
is [alto-tools](https://github.com/cneud/alto-tools) and you can check its source code for more details.

## Get the text contents of pages into the stats CSV

Now the previous output CSV becomes the input CSV:

    ./addtext.sh input.csv
    
Which will replace the last column `path` with a new one `text` and save it in 
a new `input_with_text.csv` file (also line-by-line).

## Classify extracted texts

For this step you should edit the header variables of `labgID.py`  (like upper and lower text 
perplexity thresholds, input and output files, confidence threshold for language 
identification, common languages of the input files, etc.) and then run:

    python3 langID.py

Which should create a CSV with the following columns: 

    file, page, lang, is_text_good, text

The resulting file should be filtered out based on the boolean `is_text_good` column to proceed to the next step.

When detecting text quality and language, the function may append suffixes to the base ISO language code:

 - `_trash` - Assigned if the text is likely gibberish or unusable.
   - Perplexity ≥ `PERPLEXITY_THRESHOLD_MAX`
   - Uppercase ratio > 0.9
   - Predicted language is not Latin

- `_noise` - Assigned if the text is noisy but potentially usable.
  - Perplexity ≥ `PERPLEXITY_THRESHOLD_MIN` 
  - Uppercase ratio > 0.6
  - Language is not in `COMMON_LANGS`

- `_maybe` - Assigned when the language classifier is uncertain.
  - the confidence gap between top-1 and top-2 predictions < 0.15.

- `NOISY_trash` (special case) - Returned directly if heuristic pre-checks fail (bad letter/digit/symbol/space ratios).

## Extract NER and CONLL-U of pages

Now for the CSV with at least 4 columns `file, page, lang, text` run:

    ./text_api_calls.sh inpit.csv <nametag_out_dir> <udpipe_out_dir>

Which should start filling in the output directories with predicted files: 

    <nametag_output_dir>
    ├── <file1>
        ├── <file1>-<page>.txt 
        └── ...
    ├── <file2>
        ├── <file2>-<page>.txt 
        └── ...
    └── ...
    <udpipe_output_dir>
    ├── <file1>
        ├── <file1>-<page>.txt
        └── ...
    ├── <file2>
        ├── <file2>-<page>.txt 
        └── ...
    └── ...

The configuration of the used APIs is in the `text_api_calls.sh` script, the header variables for 
output formats, default models, and other parameters are in the header.