import os 
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Header, Footer, Title,NarrativeText, ListItem
import numpy as np
import pandas as pd 
from io import StringIO
from tqdm import tqdm
# poppler_bin = r"C:\Users\user\Downloads\poppler-25.11.0\Library\bin"
poppler_bin = r"C:\Users\amith\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"

tesseract_dir = r"C:\Program Files\Tesseract-OCR"

# # ensure pdf2image/pdf->image & unstructured subprocesses find it now
os.environ["PATH"] = tesseract_dir + os.pathsep + os.environ.get("PATH", "")
# Ensure the current Python process can see it
os.environ["PATH"] = poppler_bin + os.pathsep + os.environ.get("PATH", "")



class ParsingAndChunkingHandler:
    def __init__(self, file_path, base_path):
        self.file_path = file_path
        self.base_path = base_path
        self.raw_elements = None
        self.num_raw_elements = None
        self.cleaned_elements = None
        
    
    def parse_pdf_to_elements(self):
        self.raw_elements = partition_pdf(
                                        filename=self.file_path,
                                        image_output_dir_path=self.base_path,
                                        extract_images_in_pdf=True,
                                        infer_table_structure=True,
                                        strategy="hi_res",
                                        )
        self.num_raw_elements = len(self.raw_elements)

    def clean_redundant_elements(self, text_to_ignore):
        # 3. Filter the elements
        cleaned_elements = []

        for element in self.raw_elements:
            # Check 1: Remove explicit Header/Footer types detected by the model
            if isinstance(element, (Header, Footer)):
                continue

            # Check 2: Remove elements that match our ignore list
            # We use 'strip()' to remove accidental whitespace
            clean_text = element.text.strip()

            # if clean_text in text_to_ignore:
            if np.any([i in clean_text for i in text_to_ignore]):
                continue

            # Keep the element if it passed all checks
            cleaned_elements.append(element)
        
        print(f"   -> Removed {self.num_raw_elements - len(cleaned_elements)} noise elements out of {self.num_raw_elements}.")
        self.cleaned_elements = cleaned_elements
        
        
    def find_and_chunk_title_wise(self, cleaned_elements, max_characters=11000, combine_text_under_n_chars=10, overlap=1000):
        # 4. Apply Chunking on the CLEAN list
        print("2. Chunking filtered elements...")
        chunks = chunk_by_title(
            elements=cleaned_elements,
            max_characters=max_characters,
            combine_text_under_n_chars=combine_text_under_n_chars,
            overlap = overlap
            # new_after_n_chars=4000
        )

        id_num = 0
        for c in tqdm(chunks, desc="Assigning IDs to Chunks"):
            c.metadata.id =id_num
            id_num +=1 

        return chunks
    
    def handle_table_in_chunk_for_embedding(self, element):
        """
        Combines the raw context (titles/text) with the structured table (if present).
        """
        # 1. Capture the full context (Title, Instructions, Bullet points)
        # This typically includes the section header (e.g., "Before Power Up")
        # Note: This also includes a messy version of the table, but that's okay.
        base_text = element.text.strip()

        structured_table_text = ""

        # 2. Check if there is a table and process it specifically
        if hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
            try:
                html = element.metadata.text_as_html
                # Parse HTML to DataFrame
                dfs = pd.read_html(StringIO(html))

                if dfs:
                    df = dfs[0] # Assume the first table is the main one
                    df = df.fillna('') # Clean NaNs

                    # Convert to Markdown for clear structural understanding
                    markdown_table = df.to_markdown(index=False)

                    # Create a distinct separator so the model knows this is the clean version
                    structured_table_text = f"\n\n[STRUCTURED DATATABLE]:\n{markdown_table}"

                    # OPTIONAL: Create semantic sentences for better retrieval
                    # meaningful_rows = []
                    # for _, row in df.iterrows():
                    #     meaningful_rows.append(f"Item {row.get('NAME', 'Unknown')} located in {row.get('LOCATION', 'Unknown')} must be {row.get('STATE', 'Unknown')}.")
                    # structured_table_text += "\n" + "\n".join(meaningful_rows)

            except Exception as e:
                print(f"Error parsing table HTML: {e}")

        # 3. Combine them
        # We place the Base Text first (to set context) and the Structured Table second.
        final_combined_text = f"SECTION CONTEXT:\n{base_text}{structured_table_text}"

        return final_combined_text
