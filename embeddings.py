import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
import numpy as np
import json
from IPython.display import display, Markdown

import pandas as pd
from io import StringIO
from transformers import AutoTokenizer



class EmbeddingModel:
    def __init__(self, model_name, model_dim):
        self.model_name = model_name
        self.embedding_dimension = model_dim
        self.embedding_model = HuggingFaceEmbeddings(
                                                    model_name=model_name,
                                                    model_kwargs={'device': 'cpu'} # Use 'cpu' if GPU is not available
                                                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        



    def test_chunks_tokens_and_embedding_model_tokens(self, chunks_nodes):

        max_length = self.tokenizer.model_max_length
        print(f"Analyzing {len(chunks_nodes)} nodes against model limit: {max_length} tokens")

        # 2. Collect Data
        data = []

        for i, content in enumerate(chunks_nodes):
            
            # Count tokens
            token_ids = self.tokenizer.encode(content, add_special_tokens=True)
            count = len(token_ids)
            
            # Determine status
            if count > max_length:
                status = "⚠️ TRUNCATED"
                excess = count - max_length
            else:
                status = "✅ SAFE"
                excess = 0
                
            # Append row
            data.append({
                "Node ID": i,
                "Status": status,
                "Token Count": count,
                "Excess Tokens": excess,
                "Content Snippet": content[:50].replace('\n', ' ') + "..." # One-line snippet
            })

        # 3. Create and Display Table
        df_analysis = pd.DataFrame(data)
        return df_analysis

