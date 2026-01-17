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

Each page-specific file retains the header from its original source document.

* **Input:** `../ALTO/` (input directory with ALTO XML documents)
* **Output:** `../PAGE_ALTO/` (output directory with ALTO XML files split into pages)

Example of the output directory with divided per-page XML files: [PAGE_ALTO](data_samples/PAGE_ALTO) 📁.

    PAGE_ALTO/
    ├── <file1>
        ├── <file1>-<page>.alto.xml 
        └── ...
    ├── <file2>
        ├── <file2>-<page>.alto.xml 
        └── ...
    └── ...

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

* **Input:** `../PAGE_ALTO/` (input directory with ALTO XML files split into pages from Step 1)
* **Output:** `output.csv` (table with page-level statistics and paths to ALTO files)

> [!NOTE]
> This statistics table is the basis for subsequent processing steps.
> An example is available in [test_alto_stats.csv](test_alto_stats.csv) 📎.

### ▶ Step 3: Extract text from ALTO XML

This script runs in parallel (using multiple **CPU** cores) to extract text from ALTO XMLs into `.txt` files. 
It reads the CSV from Step 2.

    python3 extract_ALTO_2_TXT.py

* **Input:** `output.csv` (from Step 2)
* **Input:** `../PAGE_ALTO/` (input directory with ALTO XML files split into pages from Step 1)
* **Output:** `../PAGE_TXT/` (directory containing raw text files)

Example of per-page text files: [PAGE_TXT](data_samples/PAGE_TXT) 📁.

    PAGE_TXT/
    ├── <file1>
        ├── <file1>-<page>.txt 
        └── ...
    ├── <file2>
        ├── <file2>-<page>.txt 
        └── ...
    └── ...

### ▶ Step 4: Classify Page Text Quality & Language

This is a key ⌛ time-consuming step that analyzes the text quality of each page, 
line-by-line, counting lines of defined types, to filter out OCR noise.

It uses the [FastText language identification model](https://huggingface.co/facebook/fasttext-language-identification) 😊 
and perplexity scores from [distilGPT2](https://huggingface.co/distilbert/distilgpt2) 😊 to detect noise.

More post-processing of TXT files can be found in the [GitHub repository](https://github.com/K4TEL/atrium-nlp-enrich.git) 
of ATRIUM project dedicated to based on
NLP enrichment of the textual data using Nametag for NER and UDPipe for CONLL-U files with lemmas & POS tags.

As the script processes, it aggregates line counts for each page into categories 🪧:

-   **Clear** - High-confidence, low-perplexity, common language.
-   **Noisy (Rough)** - Medium or Low-confidence, high-perplexity, or other OCR issues.
-   **Trash** - Hard to guess language, very high perplexity, or non-prose. 
-   **Non-text** - Failed heuristic checks (e.g., mostly digits/symbols).
-   **Empty** - Line contains only whitespace.

> [!NOTE]
> This script generates two primary output tables (results are saved every 25 pages) 
> `raw_lines_classified.csv` and `final_page_stats.csv`, while the
> raw text files and per-document results are also saved in `../PAGE-TXT/` and `../PAGE-STAT/`.

All of the input-output files and chamgable parameters are available in [config_langID.txt](config_langID.txt) 📎 where 
here Step 3 is split into three stages:

#### 4.1 Classify Lines (GPU Bound)
This script reads the extracted text files, batches lines together, and runs the FastText 
and DistilGPT2 models on the **GPU**. It logs results immediately to a raw CSV to save memory.

    python3 langID_classify.py

* **Input:** `../PAGE_TXT/` from Step 3
* **Input:** `output.csv` from Step 2
* **Output:** `DOC_LINE_LANG_CLASS/` containing per-document CSVs (e.g., [DOC_LINE_LANG_CLASS](data_samples/DOC_LINE_LANG_CLASS) 📁) like [raw_lines_classified.csv](raw_lines_classified.csv) 📎
* **Note:** This script is resume-capable. If interrupted, run it again, and it will skip files already present in the log.

`raw_lines_classified.csv>`: Page-level summary of line counts per category.
   - *Columns*:
      - `file` - document identifier
      - `page_num` - page number
      - `line_num` - line number, starts from 1 for each line on the ALTO page
      - `text` - original text of the line from ALTO page
      - `lang` - predicted ISO language code of the line ([list of all possible language labels predicted by FastText model)](https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200)
      - `lang_score` - confidence score of the predicted language code
      - `perplex` - perplexity score of the original line text
      - `categ` - assigned category of the line (**Clear**, **Noisy**, **Trash**, **Non-text**, or **Empty**)
   -   *Example*: [raw_lines_classified.csv](raw_lines_classified.csv) 📎

Example of per-document CSV file with per-line statistics: [DOC_LINE_LANG_CLASS](data_samples/DOC_LINE_LANG_CLASS) 📁.

     DOC_LINE_LANG_CLASS/
     ├── <docname1>.csv 
     ├── <docname2>.csv
     └── ...

#### 4.2 Aggregate Statistics (Memory Bound)
This script processes the directory `DOC_LINE_LANG_CLASS` with CSV files in chunks to produce the 
final page-level statistics and per-document splits (**CPU** can handle this).

    python3 langID_aggregate_STAT.py

* **Input:**  `DOC_LINE_LANG_CLASS` (directory with CSV files from previous step)
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

Example of per-document CSV file with per-page statistics: [PAGE_STAT](data_samples/PAGE_STAT) 📁.

     PAGE_STAT/
     ├── stats_<docname1>.csv 
     ├── stats_<docname2>.csv
     └── ...

### ▶ Step 5: Extract Keywords (KER) based on tf-idf

Finally, you can extract keywords 🔎 from your processed text. This script runs on a directory of subdirectories with
page-specific files `.txt`.

    python3 keywords.py -i <input_dir> -l <lang> --max-words <integer> 

-   `--input_dir`: Input directory (e.g., your output from Step 1 or text files from Step 3).
-   `--lang`: Language for KER (`cs` for Czech or `en` for English).
-   `--max-words`: Number of keywords to extract.

This process creates `.csv` table `keywords_master.csv` - the columns include `file`, and 
pairs of `keyword<N>` and `score<N>`.

An example of the summary is available in [keywords_master.csv](keywords_master.csv) 📎.

Besides the summary table, individual per-document CSV files are also created in the specified output directory

Example of per-document CSV file with keywords: [KW_PER_DOC](data_samples/KW_PER_DOC) 📁.

     KW_PER_DOC/
     ├── <docname1>.csv 
     ├── <docname2>.csv
     └── ...

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
