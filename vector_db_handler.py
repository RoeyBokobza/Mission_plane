import faiss 
import numpy as np 

class DbHandler:
    def __init__(self, embedding_module, dimension):
        self.embedding_model = embedding_module.embedding_model
        self.dimension = dimension
        self.faiss_index = None

    def build_faiss_L2_index(self, documents):
        # 1. Generate Embeddings
        print("Generating embeddings for documents...")
        doc_embeddings = self.embedding_model.embed_documents(documents)

        # 2. Convert to Numpy (Required for FAISS)
        doc_embeddings_np = np.array(doc_embeddings).astype("float32")

        # 3. Build the FAISS Index directly
        index = faiss.IndexFlatL2(self.dimension)
        index.add(doc_embeddings_np)

        print(f"FAISS Index built with {index.ntotal} documents.")
        self.faiss_index = index
        return index