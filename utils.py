
from langchain_google_genai import ChatGoogleGenerativeAI


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
