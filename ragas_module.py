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

class RagasTestingModule:
    def __init__(self, ragas_embeddings, judge_llm, vector_db_retriever):
        self.ragas_embeddings = ragas_embeddings
        self.ragas_judge_llm = judge_llm
        self.ragas_node = None
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
    


    def get_teacher_values_for_keywords_retrieval_test(self, prompt_template):

                # ==========================================
        # 3. GENERATION LOOP (The Exam - Scaled Up)
        # ==========================================
        print("\nStarting Generation Loop...")

        test_cases = []


        for i, node in enumerate(self.ragas_nodes):
            context = node.properties["page_content"]
            
            # if len(context) < 50:
            #     print(f"  - Node {i}: Skipped (Too short)")
            #     continue

            print(f"  - Processing Node {i} (Generating ~10 cases)...")

            # 1. Call Gemini
            # We increase output tokens slightly to ensure the full list fits
            response = self.ragas_judge_llm.invoke(prompt_template.format(text=context[:2000]))
            
            # 2. Clean Response
            clean_str = response.content.replace("```json", "").replace("```", "").strip()    
            # 3. Parse JSON
            data = json.loads(clean_str)
            
            # 4. Validate & Save List
            current_batch_count = 0
            if "test_cases" in data and isinstance(data["test_cases"], list):
                for item in data["test_cases"]:
                    if "query" in item and "reference" in item:
                        test_cases.append({
                            "user_input": item["query"],
                            "reference": item["reference"],
                            "source_context": context,
                            "page": node.properties["page_number"],
                            "ids": node.properties["document_id"]
                        })
                        current_batch_count += 1
                print(f"    -> Successfully added {current_batch_count} test cases.")
            else:
                print(f"    -> Failed (JSON missing 'cases' list). Raw keys: {data.keys()}")
                    
            # except Exception as e:
            #     print(f"    -> Error generating for Node {i}: {e}")
            
            # Sleep to be polite to API limits
            time.sleep(2)
            
        df_results = pd.DataFrame(test_cases)
        return df_results




    def test_keywords_retrieval_faiss(self, scenarios, k =1):
        # ==========================================
        # STEP 3: Take the Exam (Run Retrieval)
        # ==========================================
        print("Step 3: Running the Exam (Retrieving contexts for queries)...")

        # We need to add a "retrieved_contexts" column to your test dataset
        # test_questions = scenarios[scenarios['page']!=1]["user_input"].tolist()
        test_questions = scenarios["user_input"].tolist()
        ground_truths = scenarios["reference"].tolist()
        ground_truths_ids = scenarios["ids"].tolist()

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
            distances, indices = self.index.search(query_embedding_np, 1)
            
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
                "reference": ground_truths, # The correct answer/fact
                "reference_ids":ground_truths_ids, #The correct doc ID      
                "retrieved_contexts": retrieved_contexts, # What your system found
                "retrieved_ids":retrieved_ids,
                "distances": distances_found
            }

        ragas_dataset = Dataset.from_dict(evaluation_data)

        return ragas_dataset
    

    def evaluate_metrics_for_test(self, metrics, test_dataset):
        # ==========================================
        # STEP 3: Grade the Exam (Calculate Metrics)
        # ==========================================
        print("Step 3: Grading with Ragas Metrics...")


        # We assume 'ragas_llm' and 'ragas_embeddings' are already loaded from your previous code
        results = evaluate(
            dataset=test_dataset,
            metrics=metrics,
            llm=self.ragas_judge_llm,       # Use Gemini as the Judge
            embeddings=self.ragas_embeddings
        )

        # ==========================================
        # STEP 4: Show Report Card
        # ==========================================
        print("\n========================================")
        print("          EVALUATION REPORT             ")
        print("========================================")
        print(results)

        # Convert to table for detailed analysis
        df_scores = results.to_pandas()
        return df_scores
    

