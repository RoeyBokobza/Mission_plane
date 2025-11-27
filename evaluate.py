from ragas_module import *
from utils import * 
import numpy as np
import matplotlib.pyplot as plt

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
    


    def evaluate_all(self, run_cfg, k ):
        self.evaluate_classic_methods(k)
        self.evaluate_llm_as_a_judge_methods(run_cfg)

        print(f"Done Evaluation")
        return 
    
    def plot_results_graph(self):
        if not self.classic_methods_results or not self.llm_as_judge_results :
            print(f"You must run 'evaluate_all' metthod in order to plot results.")

        classic_df = pd.DataFrame.from_dict(self.classic_methods_results, orient='index')
        llm_judge_df = pd.DataFrame.from_dict(self.llm_as_judge_results, orient='index')

        combined = pd.concat([classic_df,llm_judge_df])


        names = list(combined.index)   # first column = metric names
        values = combined.iloc[:, 0].astype(float).tolist()  # second column = values

        plt.figure(figsize=(6, 4))
        bars = plt.bar(names, values, color="steelblue")

        # Add value labels on top of each bar
        for bar, v in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
            )

        plt.ylim(0, 1.1)
        plt.ylabel("Score")
        plt.title("RAG Retrieval Metrics")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()
        plt.show()



        
    



