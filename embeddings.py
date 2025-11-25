import pandas as pd
from sentence_transformers import SentenceTransformer
# import faiss
import numpy as np
import json
from IPython.display import display, Markdown

# --- 1. CONFIGURATION ---
EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
EMBEDDING_DIMENSION = 384


from transformers import AutoTokenizer

def test_chunks_tokens_and_embedding_model_tokens(chunks_nodes):
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    max_length = tokenizer.model_max_length

    print(f"Analyzing {len(chunks_nodes)} nodes against model limit: {max_length} tokens")

    # 2. Collect Data
    data = []

    for i, node in enumerate(chunks_nodes):
        content = node.properties.get('page_content', '')
        
        # Count tokens
        token_ids = tokenizer.encode(content, add_special_tokens=True)
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

    # Display the table
    print("\n--- Token Usage Summary ---")
    print(df_analysis.to_markdown(index=False))
    return df_analysis