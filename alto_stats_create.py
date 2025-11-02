import os
import argparse
import subprocess
import pandas as pd
import re

def parse_alto_tools_stats_line(line):
    """
    Parse lines like:
      # of <TextLine> elements: 33
      # of <String> elements: 163
      # of <Glyph> elements: 0
      # of <Illustration> elements: 1
      # of <GraphicalElement> elements: 10
    into a normalized dict.
    """
    m = re.match(r"# of <(\w+)> elements:\s+(\d+)", line.strip())
    if not m:
        print(line)
        return None
    element, count = m.groups()
    element = element.lower()
    mapping = {
        "textline": "textlines",
        "string": "strings",
        "glyph": "glyphs",
        "illustration": "illustrations",
        "graphicalelement": "graphics",
    }
    key = mapping.get(element, element)
    return {key: int(count)}

def run_alto_tools_stats(xml_path):
    """Run `alto-tools -s` on xml_path and return a dict of counts."""
    cmd = ["alto-tools", "-s", xml_path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error running alto-tools on {xml_path}: {e.output}")
        return None
    
    stats = {}
    for line in out.splitlines():
        parsed = parse_alto_tools_stats_line(line)
        if parsed:
            stats.update(parsed)
    return stats



def process_alto_files_with_alto_tools(directory_path):
    """Process all alto.xml files in a directory using alto-tools."""
    results = []
    for fname in os.listdir(directory_path):
        if not fname.lower().endswith(".xml"):
            continue
        xml_path = os.path.join(directory_path, fname)
        stats = run_alto_tools_stats(xml_path)
        if stats is None:
            continue
        
        # Derive file / page from filename similar to your original logic
        base = os.path.basename(fname).split(".")[0]
        parts = base.split("-")
        file_id = parts[0]
        page = parts[1] if len(parts) > 1 else ""
        
        # Build your result dictionary merging stats
        rec = {
            "file": file_id,
            "page": page,

        }
        # Map the parsed keys into your fields, e.g.:
        rec["textlines"] = int(stats.get("textlines", 0))
        rec["illustrations"] = int(stats.get("illustrations", 0))
        rec["graphics"] = int(stats.get("graphics", 0))
        # If alto-tools also gives “strings” you can capture it; else you might need fallback
        rec["strings"] = int(stats.get("strings", 0))
        rec["path"] = xml_path
        
        results.append(rec)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Folder containing ALTO XML files or subfolders with them")
    parser.add_argument("-o", "--output", default="alto_stats.csv", help="Output CSV file path")
    args = parser.parse_args()

    # Remove existing output
    if os.path.exists(args.output):
        os.remove(args.output)

    subdirs = [ os.path.join(args.input_folder, d)
                for d in os.listdir(args.input_folder)
                if os.path.isdir(os.path.join(args.input_folder, d)) ]

    first = True
    for subdir in subdirs:
        stats = process_alto_files_with_alto_tools(subdir)
        if stats:
            df = pd.DataFrame(stats)
            if first:
                df.to_csv(args.output, index=False, header=True)
                first = False
            else:
                df.to_csv(args.output, index=False, header=False, mode="a")
            print(f"Processed {len(stats)} from {subdir}")
            
    stats = process_alto_files_with_alto_tools(args.input_folder)
    if stats:
        df = pd.DataFrame(stats)
        if first:
            df.to_csv(args.output, index=False, header=True)
            first = False
        else:
            df.to_csv(args.output, index=False, header=False, mode="a")
        print(f"Processed {len(stats)} from {args.input_folder}")

    print("Done.")

if __name__ == "__main__":
    main()
