import logging
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)

logger = logging.getLogger(__name__)

def should_use_web_search(state):
    logger.info(" ROUTER: Deciding next step...")
    logger.info(f"Router: relevance_score: {state.get('relevance_score')}, has_rag_info: {state.get('has_rag_info', False)}")
    if str(state.get("relevance_score")) == "1" and state.get("has_rag_info", False):
        logger.info("Router: Returning end")
        return "end"
    logger.info("Router: Returning continue_to_web")
    return "continue_to_web"

def classify_intent(question: str) -> str:
    """Classifies user intent as diagnosis, clarification needed, or normal conversation."""
    classification_prompt = PromptTemplate(
        template="""You are an intent classifier for an automotive assistant.
        Classify the question into one of three categories: 'diagnosis_related', 'clarification_needed', or 'normal_conversation'.
        Respond with JSON: {{"intent": "<category>"}}.
        User question: "{question}" """,
        input_variables=["question"]
    )

    classifier_chain = classification_prompt | llm | JsonOutputParser()

    try:
        result = classifier_chain.invoke({"question": question})
        return result.get("intent", "normal_conversation")
    except Exception as e:
        logger.error(f"Failed to classify intent: {e}")
        return "normal_conversation"
