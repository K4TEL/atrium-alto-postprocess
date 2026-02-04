#!/usr/bin/env python3
"""
extract_LytRdr_ALTO_2_TXT.py

Step 1: Extract and reorder text from ALTO XML files using LayoutReader in parallel.
Enhanced to perform logical line segmentation and paragraph grouping based on spatial layout.

Improvements:
- Uses Vertical Overlap to robustly identify words on the same line.
- Distinguishes between standard line breaks and paragraph/block breaks.
- Preserves hyphenation fixes while maintaining layout structure.
"""

import pandas as pd
import concurrent.futures
import os
import sys
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import torch
from transformers import LayoutLMv3ForTokenClassification
import numpy as np

# --- Path Setup to find 'v3' ---
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
if str(script_dir.parent) not in sys.path:
    sys.path.append(str(script_dir.parent))

try:
    from v3.helpers import prepare_inputs, boxes2inputs, parse_logits
except ImportError:
    try:
        from layoutreader.v3.helpers import prepare_inputs, boxes2inputs, parse_logits
    except ImportError:
        print("\nCRITICAL ERROR: Could not import 'v3.helpers'.")
        sys.exit(1)

# --- Configuration ---
INPUT_CSV = "alto_statistics.csv"
OUTPUT_TEXT_DIR = "../PAGE_TXT_LR"
MAX_WORKERS = 4

# Global variables
model = None
device = None


def init_worker():
    """Initializer for worker processes."""
    global model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["transformers_verbosity"] = "error"
    try:
        model = LayoutLMv3ForTokenClassification.from_pretrained("hantian/layoutreader")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model in worker: {e}")
        sys.exit(1)


def parse_alto_xml(xml_path):
    """Parses ALTO XML to extract words and boxes."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return [], [], (0, 0)

    ns = {'alto': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

    def find_all(node, tag):
        return node.findall(f'.//alto:{tag}', ns) if ns else node.findall(f'.//{tag}')

    page = root.find('.//alto:Page', ns) if ns else root.find('.//Page')
    if page is None:
        return [], [], (0, 0)

    try:
        page_w = int(float(page.attrib.get('WIDTH')))
        page_h = int(float(page.attrib.get('HEIGHT')))
    except (ValueError, TypeError):
        return [], [], (0, 0)

    words = []
    boxes = []

    text_lines = find_all(root, 'TextLine')
    for line in text_lines:
        children = list(line)
        for i, child in enumerate(children):
            tag_name = child.tag.split('}')[-1]
            if tag_name == 'String':
                content = child.attrib.get('CONTENT')
                if not content:
                    continue
                try:
                    x = int(float(child.attrib.get('HPOS')))
                    y = int(float(child.attrib.get('VPOS')))
                    w = int(float(child.attrib.get('WIDTH')))
                    h = int(float(child.attrib.get('HEIGHT')))
                except (ValueError, TypeError):
                    continue

                # Check for explicit hyphenation tag in ALTO
                if i + 1 < len(children):
                    next_child = children[i + 1]
                    next_tag = next_child.tag.split('}')[-1]
                    if next_tag == 'HYP':
                        content += next_child.attrib.get('CONTENT', '-')

                words.append(content)
                boxes.append([x, y, x + w, y + h])

    return words, boxes, (page_w, page_h)


def normalize_boxes(boxes, width, height):
    """Normalize boxes to 0-1000 scale."""
    normalized = []
    if width == 0 or height == 0:
        return [[0, 0, 0, 0] for _ in boxes]
    x_scale = 1000.0 / width
    y_scale = 1000.0 / height
    for box in boxes:
        x1, y1, x2, y2 = box
        nx1 = max(0, min(1000, int(round(x1 * x_scale))))
        ny1 = max(0, min(1000, int(round(y1 * y_scale))))
        nx2 = max(0, min(1000, int(round(x2 * x_scale))))
        ny2 = max(0, min(1000, int(round(y2 * y_scale))))
        normalized.append([nx1, ny1, nx2, ny2])
    return normalized


def get_vertical_overlap(box1, box2):
    """
    Calculates the vertical intersection ratio between two boxes.
    Used to determine if two words are on the same visual line.
    """
    # Box format: [x1, y1, x2, y2]
    y1_a, y2_a = box1[1], box1[3]
    y1_b, y2_b = box2[1], box2[3]

    intersection = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
    min_height = min(y2_a - y1_a, y2_b - y1_b)

    if min_height <= 0:
        return 0.0

    return intersection / min_height


def post_process_text(ordered_words, ordered_boxes):
    """
    Reconstructs text from reordered words/boxes.
    Uses spatial analysis (overlap & gaps) to determine:
    1. Same line (Space)
    2. New line (Newline)
    3. New paragraph/Block (Double Newline)

    Also handles hyphenation merging.
    """
    if not ordered_words:
        return ""

    HYPHEN_CHARS = ('-', '\xad', '\u2013', '\u2014')

    # --- 1. Calculate Layout Statistics ---
    # Calculate median height to define what a "standard line" looks like.
    if ordered_boxes:
        heights = [(b[3] - b[1]) for b in ordered_boxes]
        valid_heights = [h for h in heights if h > 0]
        median_height = np.median(valid_heights) if valid_heights else 10
    else:
        median_height = 10

    # Thresholds
    # If vertical overlap > 40% of the shorter word, they are on the same line
    OVERLAP_THRESHOLD = 0.4
    # If gap > 1.8x line height, it's likely a paragraph break
    PARAGRAPH_GAP_THRESHOLD = median_height * 1.8

    result_tokens = []
    buffer_word = None

    # We need to compare current word with the *previous printed word's* box
    # to determine layout.
    prev_box = None

    for i, (word, box) in enumerate(zip(ordered_words, ordered_boxes)):

        # --- 2. Determine Separator ---
        separator = ""

        if prev_box is None:
            # First word of the file
            separator = ""
        else:
            curr_top = box[1]
            prev_bottom = prev_box[3]

            # Check for same line using vertical overlap
            overlap_ratio = get_vertical_overlap(prev_box, box)

            if overlap_ratio > OVERLAP_THRESHOLD:
                # Same visual line
                separator = " "
            else:
                # Different visual line
                vertical_gap = curr_top - prev_bottom

                if vertical_gap < -1.0 * median_height:
                    # Negative gap (significant) implies jumping to a new column/top of page
                    separator = "\n\n"
                elif vertical_gap > PARAGRAPH_GAP_THRESHOLD:
                    # Large gap implies paragraph break
                    separator = "\n\n"
                else:
                    # Standard line break
                    separator = "\n"

        # --- 3. Handle Hyphenation Buffer ---
        if buffer_word is not None:
            # We are in a merge state. The separator essentially determines
            # if we merge directly or if we need to respect a hard block break.

            # If the layout indicates a massive jump (e.g. new column), we might strictly
            # want to keep them, but standard hyphenation spans lines/cols.
            # We join them.
            current_full_word = buffer_word + word

            if current_full_word.endswith(HYPHEN_CHARS):
                # Still hyphenated chain
                buffer_word = current_full_word[:-1]
                prev_box = box  # Update position
                continue
            else:
                # Hyphenation resolved.
                # Note: We do NOT insert the calculated separator here because
                # the hyphen connects the words.
                result_tokens.append(current_full_word)
                buffer_word = None
                prev_box = box
                continue

        # --- 4. Handle Start of Hyphenation ---
        if word.endswith(HYPHEN_CHARS):
            # Apply the separator calculated for *this* word relative to the previous
            if separator == "\n\n":
                if result_tokens and result_tokens[-1] == " ": result_tokens.pop()
                result_tokens.append("\n\n")
            elif separator == "\n":
                if result_tokens and result_tokens[-1] == " ": result_tokens.pop()
                result_tokens.append("\n")
            elif separator == " ":
                if result_tokens and result_tokens[-1] not in ["\n", " "]:
                    result_tokens.append(" ")

            # Buffer the word (minus the hyphen)
            buffer_word = word[:-1]
            prev_box = box
            continue

        # --- 5. Normal Word Processing ---
        if separator == "\n\n":
            # Clean up trailing spaces before double newline
            if result_tokens and result_tokens[-1] == " ": result_tokens.pop()
            # If we just added a single newline, upgrade it; otherwise add double
            if result_tokens and result_tokens[-1] == "\n":
                result_tokens.append("\n")
            else:
                result_tokens.append("\n\n")

        elif separator == "\n":
            if result_tokens and result_tokens[-1] == " ": result_tokens.pop()
            result_tokens.append("\n")

        elif separator == " ":
            if result_tokens and result_tokens[-1] not in ["\n", " "]:
                result_tokens.append(" ")

        result_tokens.append(word)
        prev_box = box

    # Flush buffer if file ends with hyphen
    if buffer_word:
        result_tokens.append(buffer_word)

    final_text = "".join(result_tokens)

    # Cleanup strict repetitions of newlines if any
    final_text = final_text.replace("\n\n\n", "\n\n").strip()

    return final_text


def extract_single_page(args):
    """Worker function to process one page using LayoutReader."""
    file_id, page_id, xml_path_str, output_dir = args
    global model, device

    save_dir = Path(output_dir) / str(file_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"{file_id}-{page_id}.txt"

    # Skip if exists
    if txt_path.exists():
        return True

    xml_path = Path(xml_path_str)
    # Fallback logic for finding XML
    backup_xml_path = xml_path.parents[1] / "onepagers" / xml_path.name

    target_xml = xml_path
    if not target_xml.exists() and backup_xml_path.exists():
        target_xml = backup_xml_path

    if not target_xml.exists():
        return False

    try:
        words, boxes, (page_w, page_h) = parse_alto_xml(target_xml)

        if not words:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("")
            return True

        norm_boxes = normalize_boxes(boxes, page_w, page_h)
        inputs = boxes2inputs(norm_boxes)
        inputs = prepare_inputs(inputs, model)

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.cpu().squeeze(0)

        # Get the reordering indices from LayoutReader
        order_indices = parse_logits(logits, len(norm_boxes))

        # Reorder words and boxes strictly according to LayoutReader
        ordered_words = [words[i] for i in order_indices]
        ordered_boxes = [norm_boxes[i] for i in order_indices]

        # Process the layout to generate formatted text
        final_text = post_process_text(ordered_words, ordered_boxes)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        return True

    except Exception as e:
        print(f"Error processing {file_id}-{page_id}: {e}")
        return False


def main():
    if not Path(INPUT_CSV).exists():
        print(f"Error: {INPUT_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} pages to extract.")

    tasks = []
    for _, row in df.iterrows():
        tasks.append((row['file'], row['page'], row['path'], OUTPUT_TEXT_DIR))

    print(f"Extracting with {MAX_WORKERS} workers using LayoutReader...")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Process in parallel
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=init_worker
    ) as executor:
        results = list(tqdm(executor.map(extract_single_page, tasks, chunksize=1), total=len(tasks)))

    success_count = sum(results)
    print(f"Extraction complete. Success rate: {success_count / len(results):.2%}")


if __name__ == "__main__":
    main()