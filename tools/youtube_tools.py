import logging
import requests
from dotenv import load_dotenv
import os

from utils.video_utils import process_video_query
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm_vision = ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

logger = logging.getLogger(__name__)

def youtube_search_node(state):
    logger.info("➡️ In YouTube Search Node")
    search_term = state.get("dtc_code") or state["question"]
    params = {
        "part": "snippet", "q": f"{search_term} car repair diagnostic",
        "key": YOUTUBE_API_KEY, "maxResults": 1, "type": "video"
    }
    data = requests.get("https://www.googleapis.com/youtube/v3/search", params=params).json()
    videos = [{
        "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
        "title": item["snippet"]["title"]
    } for item in data.get("items", [])]
    return {**state, "youtube_results": videos}

def analyze_video_node(state):
    logger.info("➡️ In Analyze Video Node")
    if not state.get("youtube_results"):
        return {**state, "video_summary": None}
    summary = process_video_query(
        youtube_url=state["youtube_results"][0]["url"],
        prompt=state["question"],
        llm_vision=llm_vision,
        embeddings=embeddings
    )
    return {**state, "video_summary": summary}
