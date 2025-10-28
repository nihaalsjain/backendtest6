import logging
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
logger = logging.getLogger(__name__)

def tavily_search_node(state):
    logger.info("➡️ In Tavily Web Search Node")
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    query = state.get("dtc_code") or state["question"]
    response = client.search(query=f"{query} car diagnostic", max_results=3, search_depth="advanced")
    return {**state, "web_results": response.get("results", [])}

def format_results_node(state):
    logger.info("➡️ In Format Results Node")
    web_content = "\n\n".join([
        f"Title: {r.get('title','')}\nURL: {r.get('url','')}\nContent: {r.get('content','')}"
        for r in state.get("web_results", [])
    ])

    prompt = f"""You are an expert automotive assistant.
    Synthesize a comprehensive response using the provided web search results.
    QUESTION: {state['question']}\n\nWEB RESULTS:\n{web_content}"""

    final_answer = llm.invoke(prompt).content

    if state.get("web_results"):
        final_answer += "\n\n---\n**🕸️ Web Sources:**\n" + "".join([
            f"- [{r.get('title')}]({r.get('url')})\n" for r in state.get("web_results", [])
        ])
    if state.get("youtube_results"):
        final_answer += "\n**📺 Related YouTube Video:**\n" + "".join([
            f"- [{v.get('title')}]({v.get('url')})\n" for v in state.get("youtube_results", [])
        ])
    if state.get("video_summary"):
        final_answer += f"\n\n---\n**🎥 In-Depth Video Analysis:**\n{state['video_summary']}"

    return {**state, "answer": final_answer}
