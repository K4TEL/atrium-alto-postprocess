import xml.etree.ElementTree as ET
import os
import argparse

def split_alto_xml(input_file_path, output_dir):
    """
    Splits a single multi-page ALTO XML file into single-page files.

    Each new file contains the full header (Description, Styles) and the
    Layout for a single page, saved into the specified output directory.

    Args:
        input_file_path (str): The full path to the ALTO XML file.
        output_dir (str): The directory where split files will be saved.
    """
    try:
        # Register the ALTO namespace to properly query the XML
        namespace = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#'}
        ET.register_namespace('', 'http://www.loc.gov/standards/alto/ns-v3#')

        tree = ET.parse(input_file_path)
        root = tree.getroot()

        # Find the common header elements
        description = root.find('alto:Description', namespace)
        styles = root.find('alto:Styles', namespace)
        
        # Find all Page elements within the Layout
        pages = root.findall('.//alto:Page', namespace)
        
        if not pages:
            print(f"  -> No <Page> elements found. Skipping.")
            return

        # Get the base name of the input file for the output files
        base_name = os.path.splitext(os.path.basename(input_file_path))[0].replace(".alto", "")
        os.makedirs(os.path.join(output_dir, base_name), exist_ok=True)
        
        # Loop through each page, create a new XML, and save it
        for i, page in enumerate(pages, 1):
            # Use PHYSICAL_IMG_NR for the page number if available, otherwise use a counter
            page_number = page.get('PHYSICAL_IMG_NR', str(i))
            output_filename = f"{base_name}-{page_number}.alto.xml"
            output_filepath = os.path.join(output_dir, base_name, output_filename)

            # Create a new root <alto> element with the same attributes
            new_root = ET.Element(root.tag, root.attrib)

            # Append the header elements if they exist
            if description is not None:
                new_root.append(description)
            if styles is not None:
                new_root.append(styles)

            # Create a new <Layout> element and append the current page
            new_layout = ET.SubElement(new_root, 'Layout')
            new_layout.append(page)

            # Create a new tree and write it to a file
            new_tree = ET.ElementTree(new_root)
            new_tree.write(output_filepath, encoding='UTF-8', xml_declaration=True)

        print(f"  -> Successfully split into {len(pages)} page(s).")

    except ET.ParseError as e:
        print(f"  -> ERROR: Could not parse XML. {e}")
    except Exception as e:
        print(f"  -> ERROR: An unexpected error occurred: {e}")

def main():
    """
    Main function to handle command-line arguments and process files.
    """
    parser = argparse.ArgumentParser(
        description="Split multi-page ALTO XML files into single-page files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        help="Path to the directory containing ALTO XML files to process."
    )
    parser.add_argument(
        "output_dir",
        help="Path to the directory where split files will be saved. Will be created if it doesn't exist."
    )
    args = parser.parse_args()

    # --- 1. Validate input directory ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    # --- 2. Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to '{os.path.abspath(args.output_dir)}'\n")

    # --- 3. Process each XML file in the input directory ---
    for filename in sorted(os.listdir(args.input_dir)):
        if filename.lower().endswith('.xml'):
            input_file_path = os.path.join(args.input_dir, filename)
            print(f"Processing '{filename}'...")
            split_alto_xml(input_file_path, args.output_dir)

if __name__ == "__main__":
    main()
