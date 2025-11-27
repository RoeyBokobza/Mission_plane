
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import numpy as np


## TODO:: change to other types ---> ollama / local LLM.
def init_judge_llm(model_type):
    """ 
    Judge LLM Initialization
    """


    judge_llm = None

    if model_type == "Google Gemini":
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_GENAI_API_KEY")

        judge_llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_GENAI_MODEL_NAME"),
            temperature=0.1,
        )
    
    return judge_llm



def recall_at_k(true_ids, pred_ids, k):
    top_k = pred_ids[:k]
    if not true_ids:
        return 0.0
    hits = sum(1 for d in true_ids if d in top_k)
    return hits / len(true_ids)

def hitrate_at_k(true_ids, pred_ids, k):
    top_k = pred_ids[:k]
    return 1.0 if any(d in true_ids for d in top_k) else 0.0

def ndcg_at_k(true_ids, pred_ids, k):
    # binary relevance: 1 if doc in true_ids else 0
    top_k = pred_ids[:k]
    gains = [1.0 if d in true_ids else 0.0 for d in top_k]

    # DCG
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    # IDCG (ideal ranking: all relevant first)
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


matrics_mapping = {
    "Recall":recall_at_k,
    "HitRate":hitrate_at_k,
    "NCDG": ndcg_at_k
}


def compute_retrieval_metrics(
    dataset,
    metrics_fns,
    k=3,
    ref_key="reference_ids",
    pred_key="retrieved_ids",
):
    metric_values = {name: [] for name in metrics_fns}

    for row in dataset:
        true_ids = row[ref_key]
        pred_ids = row[pred_key]
        for name in metrics_fns:
            fn = matrics_mapping[name]
            metric_values[name].append(fn(true_ids, pred_ids, k))

    # average over all queries
    return {f"{name}@{k}": float(np.mean(vals)) for name, vals in metric_values.items()}