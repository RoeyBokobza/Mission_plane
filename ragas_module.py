from ragas.testset.graph import Node, NodeType, KnowledgeGraph
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall
from ragas import evaluate
import numpy as np 
import faiss 
import json 
from datasets import Dataset

import time ## needed only for Google API calls.
import pandas as pd 
from ragas import RunConfig
from tqdm import tqdm


class RagasTestingModule:
    def __init__(self, ragas_embeddings, judge_llm, vector_db_retriever):
        self.ragas_embeddings = ragas_embeddings
        self.ragas_judge_llm = judge_llm
        self.ragas_nodes = None
        self.index = vector_db_retriever
    
    
    def build_knowledge_graph(self, chunks, processed_chunks):
        ragas_nodes = []

        # We iterate through both the original chunks (for metadata) 
        # and your processed text (for content) simultaneously.
        for raw_chunk, clean_text in zip(chunks, processed_chunks):
            
            # Extract filename safely
            source_meta = raw_chunk.metadata.to_dict()
            filename = source_meta.get('filename') or source_meta.get('file_directory') or "manual.pdf"
            page_num = source_meta.get('page_number', 0)
            doc_id = source_meta.get('id', 'unknown')

            # Create the Node manually
            # This avoids the "AttributeError: from_langchain_document" bug
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": clean_text,  # This contains your "SECTION CONTEXT: ..." string
                    "filename": filename,
                    "page_number": page_num,
                    "document_id": doc_id
                }
            )
            ragas_nodes.append(node)
        
        self.ragas_nodes = ragas_nodes
        # Initialize the Graph
        kg = KnowledgeGraph(nodes=ragas_nodes)
        print("Knowledge Graph built successfully.")
        return kg
    


    
    def generate_qa_pairs(self, prompt, full_doc_text, num_pairs):
        print(f"Generating questions and answers pairs...")
        response = self.ragas_judge_llm.invoke(prompt.format(full_doc_text, num_pairs))
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        qa_pairs = data.get("qa_pairs", [])
        return qa_pairs


    def generate_evaluation_dataset(self,qa_pairs):
        final_dataset = []

        def is_quote_in_text(quote, text):
            # Simple fuzzy match (ignore case/whitespace)
            q = " ".join(quote.lower().split())
            t = " ".join(text.lower().split())

            return np.array([i in t for i in q.split(' ')]).all()


        for item in qa_pairs:
            query = item["query"]
            answer = item["answer"]
            q_type = item.get("type", "unknown")
            quotes = item.get("verbatim_quotes", [])
            
            # Ensure quotes is a list
            if isinstance(quotes, str): quotes = [quotes]
            
            # The Union Search
            matched_chunk_texts = []
            matched_chunk_ids = []
            
            for node in tqdm(self.ragas_nodes, desc="Searching for ground truth chunks ids.."):
                chunk_text = node.properties['page_content']
                
                # Does this chunk contain any of the evidence?
                is_relevant = False
                for quote in quotes:
                    if is_quote_in_text(quote, chunk_text):
                        is_relevant = True
                        break
                
                if is_relevant:
                    matched_chunk_texts.append(chunk_text)
                    matched_chunk_ids.append(node.properties["document_id"])
            
            # Only keep if we found source documents
            if matched_chunk_ids:
                final_dataset.append({
                    "user_input": query,
                    "reference": answer,
                    "ground_truth_contexts": matched_chunk_texts, # The Ground Truth Texts
                    "ground_truth_ids": matched_chunk_ids,     # The Ground Truth IDs
                    "query_type": q_type                       # 'single_hop' or 'multi_hop'
                })
                print(f"  - [{q_type}] Query: '{query[:30]}...' -> IDs: {matched_chunk_ids}")
            else:
                print(f"  - Warning: Quotes not found for query '{query[:30]}...'")

        if final_dataset:
            evaluation_df = pd.DataFrame(final_dataset)
        else:
            evaluation_df = pd.DataFrame(columns = ['user_input', 'reference', 'ground_truth_contexts', 'ground_truth_ids','query_type'])

        return evaluation_df
    



    def test_retrieval_faiss(self, scenarios, k =1):
        # ==========================================
        # STEP 3: Take the Exam (Run Retrieval)
        # ==========================================
        print("Step 3: Running the Exam (Retrieving contexts for queries)...")

        # We need to add a "retrieved_contexts" column to your test dataset
        # test_questions = scenarios[scenarios['page']!=1]["user_input"].tolist()
        test_questions = scenarios["user_input"].tolist()
        ground_truth_answers = scenarios["reference"].tolist()
        ground_truth_contexts = scenarios["ground_truth_contexts"].tolist()
        ground_truths_ids = scenarios["ground_truth_ids"].tolist()

        retrieved_contexts = []
        retrieved_ids = []
        distances_found = []

        doc_texts = [node.properties["page_content"] for node in self.ragas_nodes]
        doc_ids = [node.properties["document_id"] for node in self.ragas_nodes]

        for query in test_questions:
            # 1. Embed the query using the same embedding model
            query_embedding = self.ragas_embeddings.embed_query(query)
            query_embedding_np = np.array([query_embedding]).astype("float32")
            
            # 2. Search the FAISS index (Retrieve top 1 result)
            distances, indices = self.index.search(query_embedding_np, k)
            
            # 3. Extract the actual text based on the returned indices
            # indices[0] contains the list of IDs found for the first query
            found_texts = [doc_texts[idx] for idx in indices[0]]
            found_ids = [doc_ids[idx] for idx in indices[0]]

            dist_found = [distances[0][i] for i in range(len(distances[0]))]

            retrieved_contexts.append(found_texts)
            retrieved_ids.append(found_ids)
            distances_found.append(dist_found)  

            # Create the dataset Ragas expects
            evaluation_data = {
                "user_input": test_questions,      # What the user typed
                "ground_truth_answers": ground_truth_answers, # The correct answer/fact
                "ground_truth_context": ground_truth_contexts, # The correct answer/fact
                "reference_ids":ground_truths_ids, #The correct doc ID      
                "retrieved_contexts": retrieved_contexts, # What your system found
                "retrieved_ids":retrieved_ids,
                "distances": distances_found
            }

        ragas_dataset = Dataset.from_dict(evaluation_data)

        return ragas_dataset
    

    def evaluate_metrics_for_test(self, metrics, test_dataset, cfg):
        # ==========================================
        # STEP 3: Grade the Exam (Calculate Metrics)
        # ==========================================
        print("Step 3: Grading with Ragas Metrics...")


        # We assume 'ragas_llm' and 'ragas_embeddings' are already loaded from your previous code
        results = evaluate(
            dataset=test_dataset,
            metrics=metrics,
            llm=self.ragas_judge_llm,       # Use Gemini as the Judge
            embeddings=self.ragas_embeddings,
            run_config=cfg,
            column_map={
            "question": "user_input",
            # "answer": "ground_truth_answers",
            "contexts": "retrieved_contexts",
            "ground_truth": "ground_truth_answers" 
        }

        )

        # Convert to table for detailed analysis
        df_scores = results.to_pandas()
        return df_scores, results
    

