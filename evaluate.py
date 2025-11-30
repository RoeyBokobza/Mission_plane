from ragas_module import *
from utils import * 
import numpy as np
import matplotlib.pyplot as plt
from System_Usage import ResourceMonitor

class Evaluation:
    def __init__(self, llm_as_a_judge_methods, classic_methods, evaluation_dataset, ragas_module):
        self.llm_as_a_judge_methods = llm_as_a_judge_methods
        self.classic_methods = classic_methods
        self.evaluation_dataset = evaluation_dataset
        self.ragas_module = ragas_module

        self.llm_as_judge_results = None
        self.classic_methods_results = None


    def evaluate_llm_as_a_judge_methods(self, run_cfg):
        print(f"Evaluating LLM-as-a-Judge Methods : {[method.name for method in self.llm_as_a_judge_methods]}")
        soft_methods_df, results = self.ragas_module.evaluate_metrics_for_test(self.llm_as_a_judge_methods, self.evaluation_dataset, run_cfg )

        self.llm_as_judge_results = results.__dict__['scores'][0]

        return soft_methods_df, results 

    
     
    def evaluate_classic_methods(self, k):
        print(f"Evaluating Classic Methods: {self.classic_methods}")

        results_all = compute_retrieval_metrics(self.evaluation_dataset, self.classic_methods, k=k)
        self.classic_methods_results = results_all
        return results_all
    



    def evaluate_system_metrics(self, k):
        test_questions = self.evaluation_dataset.to_pandas()["user_input"].tolist()
        doc_texts = [node.properties["page_content"] for node in self.ragas_module.ragas_nodes]
        retrieval_times = []

        print("🚀 Starting Retrieval Evaluation...")
        
        # --- START MONITORING HERE ---
        with ResourceMonitor(interval=0.1) as monitor:
            
            for query in tqdm(test_questions, desc="Evaluating"):
                time_start = time.time()
                
                # 1. Embed
                query_embedding = self.ragas_module.ragas_embeddings.embed_query(query)
                query_embedding_np = np.array([query_embedding]).astype("float32")
                
                # 2. Search
                distances, indices = self.ragas_module.index.search(query_embedding_np, k)
                
                # (Optional) Materialize text if you need to test disk I/O too
                # found_texts = [doc_texts[idx] for idx in indices[0]]

                time_end = time.time()
                retrieval_times.append(time_end - time_start)

            # Capture stats before exiting
            resource_stats = monitor.get_stats()

        # --- END MONITORING ---

        avg_time = np.mean(retrieval_times)
        
        results_data = {
            "Avg Latency (sec)": [avg_time],
            "Avg CPU Util (%)": [resource_stats['avg_cpu_util']],
            "Max CPU Util (%)": [resource_stats['max_cpu_util']],
            "Avg GPU Util (%)": [resource_stats['avg_gpu_util']],
            "Avg GPU Mem (MB)": [resource_stats['avg_gpu_mem_mb']]
        }

        # 2. Create DataFrame
        self.system_methods_results = results_data
        return results_data


    def evaluate_all(self, run_cfg, k ):
        self.evaluate_classic_methods(k)
        self.evaluate_llm_as_a_judge_methods(run_cfg)
        self.evaluate_system_metrics(k)

        print(f"Done Evaluation")
        return 
    

    
    
    def plot_results_graph(self, save_folder):
        if not self.classic_methods_results or not self.llm_as_judge_results or not self.system_methods_results:
            print(f"You must run 'evaluate_all' method in order to plot results.")
            return

        # 1. Prepare Data
        classic_df = pd.DataFrame.from_dict(self.classic_methods_results, orient='index')
        llm_judge_df = pd.DataFrame.from_dict(self.llm_as_judge_results, orient='index')
        system_df = pd.DataFrame.from_dict(self.system_methods_results, orient='index')

        # Combine Classic and LLM Judge for the second plot
        quality_metrics_combined = pd.concat([classic_df, llm_judge_df], axis=0)

        # 2. Create Plot Layout (1 Row, 2 Columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # --- Plot 1: System Metrics ---
        names_sys = list(system_df.index)
        values_sys = system_df.iloc[:, 0].astype(float).tolist()

        bars1 = ax1.bar(names_sys, values_sys, color="mediumseagreen")
        
        # Labels for System Metrics
        for bar, v in zip(bars1, values_sys):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height(), 
                f"{v:.2f}", 
                ha="center", va="bottom"
            )
        
        ax1.set_title("System Performance (Latency & Resources)")
        ax1.set_ylabel("Value") # Units vary (sec, %, MB)
        ax1.tick_params(axis='x', rotation=20)


        # --- Plot 2: Quality Metrics (Classic + LLM Judge) ---
        names_qual = list(quality_metrics_combined.index)
        values_qual = quality_metrics_combined.iloc[:, 0].astype(float).tolist()

        bars2 = ax2.bar(names_qual, values_qual, color="steelblue")

        # Labels for Quality Metrics
        for bar, v in zip(bars2, values_qual):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height(), 
                f"{v:.2f}", 
                ha="center", va="bottom"
            )

        ax2.set_title("Retrieval Quality & Accuracy")
        ax2.set_ylim(0, 1.15) # Fixed scale for 0-1 scores
        ax2.set_ylabel("Score")
        ax2.tick_params(axis='x', rotation=20)


        ## saving        
        file_path = os.path.join(save_folder, "Retrieval_all_metrics_summary.png")
        
        # Save (bbox_inches='tight' prevents cutting off labels)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Metrics summary plot saved to: {file_path}")
        # Final Layout Adjustments
        plt.tight_layout()
        plt.show()




        
    



