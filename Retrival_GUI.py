import streamlit as st
import numpy as np
from ragas_module import RagasTestingModule # Import your actual class
import faiss
from ragas.testset.graph import KnowledgeGraph
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

import config



# 1. SETUP
@st.cache_resource
def get_rag_module():
    
    """ 
    Judge LLM Initialization
    """
    print(f"asdasdkfasjfkldjfml,dsjgkldsjflasddflsa")
    judge_llm = ChatNVIDIA(
        model="mistralai/mixtral-8x22b-instruct-v0.1",
        api_key="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",
        temperature=0.1,
        max_completion_tokens = 8000
    )
    ragas_llm = judge_llm
    embedding_model = HuggingFaceEmbeddings(
                                    model_name="BAAI/bge-small-en-v1.5",
                                    model_kwargs={'device': 'cpu'} # Use 'cpu' if GPU is not available
                                                )

    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# This recreates the index object in memory
    loaded_index = faiss.read_index("my_rag_index.faiss")

    rag_module = RagasTestingModule(ragas_embeddings, ragas_llm, loaded_index)
    return rag_module 






print(f"starting RAGAS FAISS Inspector GUI...")

rag_tester = get_rag_module()
print(f"RAG Module: {rag_tester}")
loaded_kg = KnowledgeGraph.load("my_ragas_graph.json")
print(f"loaded KG : {loaded_kg}")
rag_tester.ragas_nodes = loaded_kg.nodes
print("RAG Module loaded successfully.")
# except NameError:
#     st.error("RAG Module not loaded. Please initialize 'rag_tester' in your code.")
#     st.stop()
# except Exception as e :
#     st.error(f"An error occurred: {e}")
#     st.stop()

all_chunks = [node.properties["page_content"] for node in rag_tester.ragas_nodes]

# 3. GUI LAYOUT
st.title("🔎 Raw FAISS Inspector")

# Input
query = st.text_input("Enter test query:", "What are the safety protocols?")
k_val = st.slider("Number of chunks (k):", min_value=1, max_value=10, value=3)

# 4. SEARCH LOGIC
if query and st.button("Search Index"):
    with st.spinner("Embedding & Searching..."):
        
        # A. Embed the query
        # We need a numpy array of shape (1, dimension). 
        # Adjust 'embed_query' to whatever method your embedding model uses.
        query_vector = rag_tester.ragas_embeddings.embed_query(query)
        query_embedding_np = np.array([query_vector]).astype('float32')
        
        # B. Your Raw FAISS Search
        # distances and indices are shape (1, k) because we sent 1 query
        distances, indices = rag_tester.index.search(query_embedding_np, k_val)
        
        # C. Process Results
        # We access [0] because FAISS returns a batch of results
        found_indices = indices[0]
        found_distances = distances[0]

    # 5. DISPLAY
    st.write(f"**Top {k_val} Results:**")
    
    for i, (idx, dist) in enumerate(zip(found_indices, found_distances)):
        # Handle case where FAISS returns -1 (meaning not enough neighbors found)
        if idx == -1: 
            continue
            
        # D. RETRIEVE TEXT (The Crucial Step)
        # You must map the integer 'idx' to your actual text list
        try:
            # Replace 'all_chunks' with your actual list of strings/nodes
            retrieved_text = all_chunks[idx]
        except IndexError:
            retrieved_text = "Error: Index out of bounds in text store."
            
        with st.expander(f"Rank {i+1} | ID: {idx} | L2 Dist: {dist:.4f}"):
            st.markdown(f"**Content:**")
            st.text(retrieved_text)