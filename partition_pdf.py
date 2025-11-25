import os 
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Header, Footer, Title,NarrativeText, ListItem
import numpy as np

poppler_bin = r"C:\Users\user\Downloads\poppler-25.11.0\Library\bin"

tesseract_dir = r"C:\Program Files\Tesseract-OCR"

# ensure pdf2image/pdf->image & unstructured subprocesses find it now
os.environ["PATH"] = tesseract_dir + os.pathsep + os.environ.get("PATH", "")
# Ensure the current Python process can see it
os.environ["PATH"] = poppler_bin + os.pathsep + os.environ.get("PATH", "")



class chunking_procedure:
    def __init__(self, file_path, base_path):
        self.file_path = file_path
        self.base_path = base_path
        self.raw_elements = partition_pdf(
                                        filename=self.full_path,
                                        image_output_dir_path=self.base_path,
                                        extract_images_in_pdf=True,
                                        infer_table_structure=True,
                                        strategy="hi_res",
                                        )
        self.num_raw_elements = len(self.raw_elements)
        self.cleaned_elements = None
        
    
    def clean_redundant_elements(self, text_to_ignore):
        # 3. Filter the elements
        cleaned_elements = []

        original_count = len(self.raw_elements)

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
        
        print(f"   -> Removed {original_count - len(cleaned_elements)} noise elements.")
        self.cleaned_elements = cleaned_elements
        
        
    def find_and_chunk_title_wise(self, cleaned_elements):
        # 4. Apply Chunking on the CLEAN list
        print("2. Chunking filtered elements...")
        chunks = chunk_by_title(
            elements=cleaned_elements,
            max_characters=11000,
            combine_text_under_n_chars=10,
            overlap = 1000
            # new_after_n_chars=4000
        )
        return chunks
    
