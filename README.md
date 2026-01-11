# 📦 ALTO XML Files Postprocessing Pipeline

This project provides a complete workflow for processing ALTO XML files. It takes raw ALTO 
XMLs and transforms them into structured statistics tables, performs text classification, 
filters low-quality OCR results, and extracts high-level linguistic features like 
Named Entities (NER), CONLL-U files with lemmas & part-of-sentence tags, and keywords (KER).

The core of the quality filtering relies on language identification and perplexity measures 
to identify and categorize noisy or unreliable OCR output.

---

## ⚙️ Setup

Before you begin, set up your environment.

1.  Create and activate a new virtual environment in the project directory 🖥.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Clone and install `alto-tools` 🔧, which is used for statistics and text extraction:
    ```bash
    git clone https://github.com/cneud/alto-tools.git
    cd alto-tools
    pip install .
    cd .. 
    ```
4.  Download the FastText model 😊 for language identification:
    ```bash
    wget (https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin) -O lid.176.bin
    ```
You are now ready to start the workflow.

---

## Workflow Stages

The process is divided into sequential steps, starting from raw ALTO files and ending 
with extracted linguistic and statistic data.

### ▶ Step 1: Split Document-Specific ALTOs into Pages

First, ensure you have a directory 📁 containing your document-level `<file>.alto.xml` files. 
This script will split them into individual page-specific XML files.

    python3 page_split.py <input_dir> <output_dir>

This command will generate a new directory structure: 

    <output_dir>
    ├── <file1>
        ├── <file1>-<page>.alto.xml 
        └── ...
    ├── <file2>
        ├── <file2>-<page>.alto.xml 
        └── ...
    └── ...

Each page-specific file retains the header from its original source document.

### ▶ Step 2: Create Page Statistics Table

Next, use the output directory from Step 1 as the input for this script to generate a 
foundational CSV statistics file.

    python3 alto_stats_create.py <input_dir> -o output.csv

This script writes a CSV file line-by-line, capturing metadata for each page:

    file, page, textlines, illustrations, graphics, strings, path
    CTX200205348, 1, 33, 1, 10, 163, /lnet/.../A-PAGE/CTX200205348/CTX200205348-1.alto.xml
    CTX200205348, 2, 0, 1, 12, 0, /lnet/.../A-PAGE/CTX200205348/CTX200205348-2.alto.xml
    ...

The extraction is powered by the [alto-tools](https://github.com/cneud/alto-tools) 🔗 framework.

> [!NOTE]
> This statistics table is the basis for subsequent processing steps.
> An example is available in [test_alto_stats.csv](test_alto_stats.csv) 📎.

### ▶ Step 3: Classify Page Text Quality & Language

This is a key ⌛ time-consuming step that analyzes the text quality of each page, 
line-by-line, counting lines of defined types, to filter out OCR noise.

It uses the [FastText language identification model](https://huggingface.co/facebook/fasttext-language-identification) 😊, 
autocorrection libraries ([pyspellchecker](https://pypi.org/project/pyspellchecker/) 🔗,
[autocorrect](https://github.com/filyp/autocorrect) 🔗), and perplexity scores 
from [distilGPT2](https://huggingface.co/distilbert/distilgpt2) 😊 to detect noise.

As the script processes, it aggregates line counts for each page into categories 🪧:

-   **Clear** - High-confidence, low-perplexity, common language.
-   **Rough** - Medium-confidence, but still likely a common language.
-   **Noisy** - Low-confidence, high-perplexity, or other OCR issues.
-   **Trash** - Hard to guess language, very high perplexity, or non-prose. 
-   **Short** - Too few words to classify confidently.
-   **Non-text** - Failed heuristic checks (e.g., mostly digits/symbols).
-   **Empty** - Line contains only whitespace.
-   **N/A** - Used internally for error cases.


> [!NOTE]
> This script generates two primary output tables (results are saved every 25 pages) 
> `raw_lines_classified.csv` and `final_page_stats.csv`, while the
> raw text files and per-document results are also saved in `../PAGE-TXT/` and `../PAGE-STAT/`.

All of the input-output files and chamgable parameters are available in [config_langID.txt](config_langID.txt) 📎 where 
here Step 3 is split into three stages:

#### 3.1 Extract Text (CPU Bound)
This script runs in parallel (using multiple **CPU** cores) to extract text from ALTO XMLs into `.txt` files. It reads the CSV from Step 2.

    python3 langID_extract_TXT.py

* **Input:** `output.csv` (from Step 2)
* **Output:** `../PAGE_TXT/` (directory containing raw text files)

#### 3.2 Classify Lines (GPU Bound)
This script reads the extracted text files, batches lines together, and runs the FastText 
and DistilGPT2 models on the **GPU**. It logs results immediately to a raw CSV to save memory.

    python3 langID_classify.py

* **Input:** `../PAGE_TXT/` and `output.csv`
* **Output:** `raw_lines_classified.csv` (append-only log of every line)
* **Note:** This script is resume-capable. If interrupted, run it again, and it will skip files already present in the log.


`raw_lines_classified.csv>`: Page-level summary of line counts per category.
   - *Columns*:
      - `file` - document identifier
      - `page` - page number
      - `line_num` - line number, starts from 1 for each line on the ALTO page
      - `text` - original text of the line from ALTO page
      - `lang` - predicted ISO language code of the line ([list of all possible language labels predicted by FastText model)](https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200)
      - `score` - confidence score of the predicted language code
      - `ppl` - perplexity score of the original line text
      - `cat` - assigned category of the line (**Clear**, **Noisy**, **Trash**, **Non-text**, or **Empty**)
   -   *Example*: [raw_lines_classified.csv](raw_lines_classified.csv) 📎


#### 3.3 Aggregate Statistics (Memory Bound)
This script processes the massive `raw_lines_classified.csv` in chunks to produce the 
final page-level statistics and per-document splits (**CPU** can handle this).

    python3 langID_aggregate_STAT.py

* **Input:** `raw_lines_classified.csv`
* **Output 1:** `final_page_stats.csv` (The input CSV augmented with line counts: `clear_lines`, `noisy_lines`, etc.)
* **Output 2:** `../PAGE_STAT/` (Folder containing per-document CSVs)

`final_page_stats.csv`: Detailed classification results for *every single line*.
   - *Columns*:
      - `file` - document identifier
      - `page` - page number
      - `Clear` - clear lines count, clean and ready to be processed
      - `Non-text` - non-text lines count, contain mostly digits/symbols
      - `Trash` - trash lines count, unintelligible or very high perplexity (due to OCR errors)
      - `Noisy` - noisy lines count, some errors but partially understandable
      - `Empty` - empty lines count, contain only whitespace
   -   *Example*: [final_page_stats.csv](final_page_stats.csv) 📎


### ▶ Step 4: Add Full Text Content to Statistics CSV

After classification, you may want a CSV that includes the full, concatenated text for pages 
deemed high-quality (or all pages). This script adds text content directly into your 
statistics CSV.

    ./addtext.sh input.csv

This will take `input.csv` (e.g., your `line_counts_...csv` from Step 3), replace 
the `path` column with a new `text` column, and save the result as `input_with_text.csv`.

### ▶ Step 5: Extract NER and CONLL-U

Using a CSV file that contains text content (like the one from Step 4), you can 
now call external APIs to perform advanced NLP analysis.

The input CSV must have at least these 4 columns: `file, page, lang, text`.

    ./text_api_calls.sh input.csv <nametag_out_dir> <udpipe_out_dir>

This script will populate the output directories 📁 with `.txt` files containing the 
API results for Name Entity Recognition (NER) and Universal Dependencies (CONLL-U).

    <nametag_output_dir>
    ├── <file1>
        ├── <file1>-<page>.txt 
        └── ...
    <udpipe_output_dir>
    ├── <file1>
        ├── <file1>-<page>.txt
        └── ...

You can configure ⚙️ API endpoints, models, and other parameters by editing the 
header variables in the `text_api_calls.sh` script.

### ▶ Step 6: Extract Keywords (KER) based on tf-idf

Finally, you can extract keywords 🔎 from your processed text. This script runs on a directory 
of page-specific files, either `.alto.xml` or `.txt`.

    python3 run_ker.py --dir <input_dir> --lang <lang> --max-words <integer> --file-format <file_format>

-   `--dir`: Input directory (e.g., your output from Step 1 or text files from Step 3).
-   `--lang`: Language for KER (`cs` for Czech or `en` for English).
-   `--max-words`: Number of keywords to extract.
-   `--file-format`: `alto` or `txt`.

This process creates two `.tsv` tables:

1.  `pages_keywords.tsv`: Keywords extracted per-page.
2.  `documents_keywords.tsv`: Summarized keywords for each document.

The columns include `file`, `page` (for the per-page table), `lang`, `threshold`, and pairs of `keyword<N>` and `score<N>`.

An example of the per-document summary is available in [documents_keywords.tsv](documents_keywords.tsv) 📎.

---

## Acknowledgements 🙏

**For support write to:** lutsai.k@gmail.com responsible for this GitHub repository [^8] 🔗

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4]  💰
- **Shared by** ATRIUM [^4] & UFAL [^7] 🔗

**©️ 2025 UFAL & ATRIUM**

[^4]: https://atrium-research.eu/
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^7]: https://ufal.mff.cuni.cz/home-page
