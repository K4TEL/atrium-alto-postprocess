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


### ▶ Step 4: Extract NER and CONLL-U

This stage performs advanced NLP analysis using external APIs (Lindat/CLARIAH-CZ) to generate Universal Dependencies (CoNLL-U) and Named Entity Recognition (NER) data.

Unlike previous steps, this process is split into modular shell scripts to handle large-scale processing, text chunking, and API rate limiting.

#### 5.1 Configuration ⚙️

Before running the pipeline, review the `api_config.env` file. This file controls directory paths, API endpoints, and model selection.
```bash
# Example settings in api_config.env
INPUT_DIR="../PAGE_TXT"        # Source of text files (from Step 3.1)
OUTPUT_DIR="../OUT_API"        # Destination for results
MODEL_UDPIPE="czech-pdt-ud-2.15-241121"
MODEL_NAMETAG="nametag3-czech-cnec2.0-240830"
WORD_CHUNK_LIMIT=900           # Word limit per API call
```

#### 5.2 Execution Pipeline

Run the following scripts in sequence. Each script utilizes [api_common.sh](api_util/api_common.sh) for logging, retry logic, and error handling.

##### I. Generate Manifest

Maps input text files to document IDs and page numbers to ensure correct processing order.
```bash
./api_manifest.sh
```

* **Input:** `INPUT_DIR` (raw text files in subdirectories).
* **Output:** `processing_work/manifest.tsv`.

##### II. UDPipe Processing (Morphology & Syntax)

Sends text to the UDPipe API. Large pages are automatically split into chunks (default 900 words) using [chunk.py](api_util/chunk.py) to respect API limits, then merged back into valid CoNLL-U files.
```bash
./api_udp.sh
```

* **Output:** `processing_work/UDPIPE_INTERMEDIATE/*.conllu` (Intermediate CoNLL-U files).

##### III. NameTag Processing (NER)

Takes the valid CoNLL-U files and passes them through the NameTag API to annotate Named Entities 
(NE) directly into the syntax trees.
```bash
./api_nt.sh
```

* **Output:** `OUTPUT_DIR/CONLLU_FINAL/` (Final annotated files).

##### IV. Generate Statistics

Aggregates the entity counts from the final CoNLL-U files into a summary CSV. It utilizes [analyze.py](api_util/analyze.py) to map complex 
CNEC 2.0 tags (e.g., `g`, `pf`, `if`) into human-readable categories (e.g., "Geographical name", "First name", "Company/Firm").

```bash
./api_stats.sh
```

* **Output:** `OUTPUT_DIR/STATS/summary_ne_counts.csv`.

#### 5.3 Output Structure

After completing the pipeline, your output directory will be organized as follows:
```
processing_work/
├── UDPIPE_INTERMEDIATE/  # Intermediate CONLL-U files
│   ├── <doc_id>_part1.conllu
│   ├── <doc_id>_part2.conllu
│   └── ...
├── nametag_response_docname1.conllu.json
├── nametag_response_docname2.conllu.json
├── ...
└── manifest.tsv
```
AND
```
<OUTPUT_DIR>
├── CONLLU_FINAL/           # Full linguistic analysis
│   ├── <doc_id>.conllu     # Parsed sentences with NER tags
│   └── ...
└── STATS/
    └── summary_ne_counts.csv  # Table of top entities per document
```

### ▶ Step 5: Extract Keywords (KER) based on tf-idf

Finally, you can extract keywords 🔎 from your processed text. This script runs on a directory of subdirectories with
page-specific files `.txt`.

    python3 keywords.py -i <input_dir> -l <lang> --max-words <integer> 

-   `--input_dir`: Input directory (e.g., your output from Step 1 or text files from Step 3).
-   `--lang`: Language for KER (`cs` for Czech or `en` for English).
-   `--max-words`: Number of keywords to extract.

This process creates `.csv` table `keywords_master.csv`

The columns include `file`, and pairs of `keyword<N>` and `score<N>`.

An example of the summary is available in [keywords_master.csv](keywords_master.csv) 📎.

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
