EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

EMBEDDING_DIMENSION = 384


LLM_JUDGE_MODEL_TYPE = "Google Gemini"
## Google Gemini Model as LLM as a Judge

GOOGLE_GENAI_API_KEY = "AIzaSyCrC4ep_YrCUrCMQLFvudbfnh-tgIMW62A"
GOOGLE_GENAI_MODEL_NAME = "gemini-2.5-flash"

SINGLE_HOP_NUM_PAIRS = 10
MULTI_HOP_NUM_PAIRS = 10

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
                                "verbatim_quotes": [
                                    "Quote supporting part A",
                                    "Quote supporting part B"
                                ]
                            }},
                            ...
                        ]
                    }}
                    """