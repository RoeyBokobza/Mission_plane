""" 
Trial Running Folder
"""
TRIAL_FOLDER_NAME = "/Results/Trial_1"
TRIAL_DESC = "Initial Trial with BGE Small Embeddings and Nvidia LLM Judge. In this trial there is no table handling, " \
"no chunking by title, everything as naive as it can be. Using FAISS L2 as vector DB."




""" 
Embedding Model
"""
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

"""
DB config
"""
NUM_DOCS_TO_RETRIEVE = 1

""" 
Judge LLM config
"""
# LLM_JUDGE_MODEL_TYPE = "Google Gemini"
LLM_JUDGE_MODEL_TYPE = "Nvidia"

GOOGLE_GENAI_API_KEY = "AIzaSyCrC4ep_YrCUrCMQLFvudbfnh-tgIMW62A"
NVIDIA_API_KEY = "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"

# MODEL_NAME = "gemini-2.5-flash"
MODEL_NAME = "mistralai/mixtral-8x22b-instruct-v0.1"

""" 
Eval Dataset Creation Prompts.
"""

SINGLE_HOP_NUM_PAIRS = 50
MULTI_HOP_NUM_PAIRS = 50

""" 
Prompts Configuration
"""

SINGLE_HOP_PROMPT = """
                    You are an expert creating an evaluation dataset for a RAG system.
                    I have provided the text of a technical manual below.

                    TEXT:
                    {} 

                    INSTRUCTIONS:
                    1. Generate {} search queries. 
                    2. Queries must be "Single-Hop" (Answer found in one specific section).
                    2. Provide the comprehensive answer for each.
                    3. CRITICAL: Extract a LIST of "verbatim_quotes" (short text snippets) that support the answer. 

                    OUTPUT FORMAT (JSON List):
                    {{
                        "qa_pairs": [
                            {{
                                "query": "Simple query about X...",
                                "answer": "Answer about X.",
                                "type": "single_hop",
                                "verbatim_quotes": ["Quote about X"]
                            }}
                            ]
                    }}
                    """

MULTI_HOP_PROMPT = """
                    You are an expert creating an evaluation dataset for a RAG system.
                    I have provided the text of a technical manual below.

                    TEXT:
                    {}


                    INSTRUCTIONS:
                    1. Generate {} "Multi-Hop" search queries. 
                    - A Multi-Hop query requires information from AT LEAST TWO different sections/pages to answer (the more the better).
                    - Example: "What is the voltage for EFB (Page 1) and what is the safety warning for it (Page 5)?"
                    2. Provide the comprehensive answer.
                    3. CRITICAL: Extract a LIST of "verbatim_quotes" (short text snippets) that support the answer. 
                    - You MUST include at least one quote from each relevant section.

                    OUTPUT FORMAT (JSON List):
                    {{
                        "qa_pairs": [
                            {{
                                "query": "Complex query connecting A and B...",
                                "answer": "The comprehensive answer combining A and B.",
                                "type": "multi_hop",
                                "verbatim_quotes": [
                                    "Quote supporting part A",
                                    "Quote supporting part B"
                                ]
                            }},
                            ...
                        ]
                    }}
                    """