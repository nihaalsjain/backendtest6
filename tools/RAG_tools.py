"""
RAG tools for vehicle assistant - Complete integration from version3_refactor.py
This module contains the @tool-decorated helper functions used by the diagnostic workflow.
"""

import os
import re
import json
import logging
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global variable to store latest diagnostic data for dual-channel approach
_latest_diagnostic_data = None

# -------------------------
# Paths and Setup
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DB_DIR = BASE_DIR / "output" / "chroma_db"

if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
    logger.error("❌ Chroma DB not found. Please run RAG setup first.")
    # Don't exit, just log error - let the tools handle gracefully

# -------------------------
# Initialize LLM + embeddings + retriever
# -------------------------
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing - set it in .env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

try:
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings,
        collection_name="pdf_chunks",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
except Exception as e:
    logger.error(f"Failed to initialize Chroma vectorstore: {e}")
    vectorstore = None
    retriever = None

# Retrieval grader setup
grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
grader_prompt = PromptTemplate(
    template="""You are a teacher grading a quiz. You will be given: 1/ a QUESTION 2/ A FACT provided by the student
You are grading RELEVANCE RECALL: A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
Question: {question} Fact: {documents}""",
    input_variables=["question", "documents"],
)
retrieval_grader = grader_prompt | grader_llm | JsonOutputParser()

# Global NLP model (lazy-loaded)
_nlp_model = None

def _get_nlp():
    global _nlp_model
    if _nlp_model is None:
        import spacy
        try:
            _nlp_model = spacy.load("en_core_web_trf")
        except Exception:
            try:
                _nlp_model = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Could not load spacy model: {e}")
                _nlp_model = None
    return _nlp_model

# -------------------------
# TOOL FUNCTIONS (Complete from version3_refactor.py)
# -------------------------

@tool
def enhance_question_with_vehicle(question: str, vehicle_info: str = None, thread_id: str = "default") -> dict:
    """Enhance the user's question by incorporating vehicle information."""
    
    logger.info(f"🔍 RAG TOOL CALLED: enhance_question_with_vehicle")
    logger.info(f"🔍 Original question: {question}")
    logger.info(f"🔍 Vehicle info: {vehicle_info}")
    logger.info(f"🔍 Thread ID: {thread_id}")
    
    # Print vehicle info being used for this query
    print(f"🚗 VEHICLE INFO USED: {vehicle_info if vehicle_info else 'None'}")
    
    if not vehicle_info:
        return {
            "enhanced_question": question,
            "vehicle_added": False,
            "message": "No vehicle information to add"
        }
    
    # Check if question already contains the vehicle info
    question_lower = question.lower()
    vehicle_lower = vehicle_info.lower()
    
    if vehicle_lower in question_lower:
        logger.info("Question already contains vehicle information")
        return {
            "enhanced_question": question,
            "vehicle_added": False,
            "message": "Question already contains vehicle information"
        }
    
    # Don't enhance if it's a DTC code question
    if re.search(r'\b[PB]\d{4}\b', question):
        logger.info("Question contains DTC code, not enhancing")
        return {
            "enhanced_question": question,
            "vehicle_added": False,
            "message": "DTC question - no enhancement needed"
        }
    
    # Try to naturally incorporate vehicle info
    enhanced_question = question
    vehicle_added = False
    
    if "how" in question_lower and any(word in question_lower for word in ['change', 'replace', 'fix', 'repair']):
        # Transform "how can i change brake pads" → "how can i change brake pads of Honda Civic"
        enhanced_question = question.rstrip('?') + f" of {vehicle_info}?"
        vehicle_added = True
    elif "what" in question_lower:
        # Transform "what is wrong with my engine" → "what is wrong with my Honda Civic engine"
        enhanced_question = question.replace("my ", f"my {vehicle_info} ")
        vehicle_added = True
    else:
        # Generic enhancement - add vehicle at the beginning
        enhanced_question = f"{vehicle_info}: {question}"
        vehicle_added = True
    
    logger.info(f"Enhanced question: {enhanced_question}")
    
    return {
        "enhanced_question": enhanced_question,
        "vehicle_added": vehicle_added,
        "message": f"Enhanced question with {vehicle_info}"
    }


@tool
def is_vehicle_related(question: str) -> dict:
    """Check if the question is vehicle-related before processing."""
    classifier_prompt = f"""
You are a classifier. Decide if the user question is about vehicle diagnostics, repair, or automotive problems.
Answer YES if it's about vehicle issues, maintenance, repairs, faults, or checks.
Answer NO if it's unrelated to vehicles.
Examples:
- "How do I replace my brake pads?" → YES
- "What is the capital of France?" → NO
- "Can I change the engine oil myself?" → YES
- "Tell me a joke." → NO
QUESTION: {question}
Answer only with "YES" or "NO".
"""
    try:
        response = llm.invoke(classifier_prompt)
        is_related = response.content.strip().upper() == "YES"
        logger.info(f"Vehicle relation check: {question[:50]}... → {is_related}")
        return {
            "is_vehicle_related": is_related,
            "message": "Vehicle-related question detected" if is_related else "Not vehicle-related",
        }
    except Exception as e:
        logger.error(f"Error in vehicle relation check: {e}")
        return {"is_vehicle_related": True, "message": "Error in classification, defaulting to vehicle-related"}

@tool
def extract_vehicle_model(question: str) -> dict:
    """Extract vehicle make and model from the question using NLP."""
    nlp_model = _get_nlp()
    if not nlp_model:
        logger.warning("NLP model not available, falling back to simple extraction")
        # Simple fallback extraction
        words = question.split()
        potential_vehicle = []
        for i, word in enumerate(words):
            if word.lower() in ['toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'volkswagen', 'nissan', 'hyundai', 'kia']:
                potential_vehicle.append(word.title())
                if i + 1 < len(words):
                    potential_vehicle.append(words[i + 1].title())
                break
        
        if potential_vehicle:
            vehicle_info = " ".join(potential_vehicle[:2])
            return {"vehicle_info": vehicle_info, "found": True}
        else:
            return {"vehicle_info": None, "found": False}
    
    doc = nlp_model(question)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            model_tokens = [ent.text]
            next_token = ent.end
            while next_token < len(doc) and (
                doc[next_token].is_title or doc[next_token].like_num or doc[next_token].is_lower
            ):
                model_tokens.append(doc[next_token].text)
                next_token += 1
            entities.append(" ".join(model_tokens))

    if entities:
        vehicle_info = entities[0].title()
    else:
        tokens = [t for t in doc if not t.is_stop and t.pos_ in ["PROPN", "NUM", "NOUN"]]
        if len(tokens) >= 2:
            model_tokens = []
            for t in tokens:
                if t.pos_ in ["PROPN", "NUM"]:
                    model_tokens.append(t.text)
                else:
                    break
            vehicle_info = " ".join(model_tokens).title() if model_tokens else None
        else:
            vehicle_info = None

    logger.info(f"Vehicle extraction: {question[:50]}... → {vehicle_info}")
    return {"vehicle_info": vehicle_info, "found": vehicle_info is not None}

def normalize_dtc_codes(text: str) -> str:
    """
    Normalize spoken DTC codes to standard format - ULTRA FLEXIBLE VERSION.
    
    Examples:
    - "P zero three zero one" → "P0301"
    - "tell me about p zero three zero one" → "tell me about P0301"
    """
    
    # Dictionary to convert word numbers to digits
    word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    
    # ULTRA FLEXIBLE pattern - allows any non-word chars between parts
    pattern = r'\b([PBCUpbcu])\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\b'
    
    def replace_dtc(match):
        prefix = match.group(1).upper()  # P, B, C, or U
        digit1 = word_to_digit[match.group(2).lower()]
        digit2 = word_to_digit[match.group(3).lower()]
        digit3 = word_to_digit[match.group(4).lower()]
        digit4 = word_to_digit[match.group(5).lower()]
        
        result = f"{prefix}{digit1}{digit2}{digit3}{digit4}"
        logger.info(f"🔧 DTC Match found: '{match.group(0)}' → '{result}'")
        return result
    
    # Apply the replacement
    normalized_text = re.sub(pattern, replace_dtc, text, flags=re.IGNORECASE)
    return normalized_text

@tool
def search_vehicle_documents(question: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search vehicle diagnostic documents for relevant information."""
    question = normalize_dtc_codes(question)
    logger.info(f"🔍 Searching documents for: {question}") 
    

    if not retriever:
        logger.error("Retriever not available")
        return {
            "answer": "Document search not available - database not initialized.",
            "source_documents": [],
            "has_rag_info": False,
            "dtc_code": dtc_code,
            "vehicle_info": vehicle_info,
            "selected_chunk_label": "ERROR",
            "selected_chunk_content": ""
        }
    
    # Check for DTC code in question
    dtc_match = re.search(r"\b([PBUC]\d{4})\b", question.upper())
    if dtc_match:
        dtc_code = dtc_match.group(1)
    
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        
        if not docs:
            return {
                "answer": "No relevant information found in the PDF.",
                "source_documents": [],
                "has_rag_info": False,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "NONE",
                "selected_chunk_content": ""
            }
        
        print(f"📊 Retrieved {len(docs)} documents from vector store")

        # Display retrieved chunks with media information (like rag_test_basic)
        print("\n🔎 Retrieved Chunks:")
        for i, d in enumerate(docs, 1):
            pages = d.metadata.get("pages") or d.metadata.get("page") or "?"
            snippet = d.page_content[:300].replace('\n', ' ') + ('...' if len(d.page_content) > 300 else '')
            
            # Check for media using the same function as rag_test_basic
            media_info = extract_media_references_enhanced(d.page_content)
            media_indicators = []
            if media_info['images']:
                media_indicators.append(f"📷{len(media_info['images'])}")
            if media_info['tables']:
                media_indicators.append(f"📊{len(media_info['tables'])}")
            
            media_str = f" [{', '.join(media_indicators)}]" if media_indicators else ""
            
            print(f"  {i}. pages={pages} chars={len(d.page_content)}{media_str}")
            print(f"     snippet={snippet}")

        # Build context using the SAME format as rag_test_basic.py
        blocks = []
        for i, d in enumerate(docs, 1):
            pages = d.metadata.get("pages") or d.metadata.get("page") or "?"
            
            # Format content with media information (like rag_test_basic)
            formatted_content = format_content_with_media_enhanced(d.page_content, i)
            
            blocks.append(f"[DOC {i} | pages: {pages}]\n{formatted_content}")
        
        context = "\n\n".join(blocks)
        
        # Use the SAME prompt structure as rag_test_basic.py
        prompt = (
            "You are a helpful assistant. Answer the question ONLY using the provided context. "
            "When referencing images, figures, or tables mentioned in the context, include them in your response. "
            "If the answer is not present, reply exactly: I don't know.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        
        response = llm.invoke(prompt)
        answer_text = response.content.strip()
        print(f"🤖 Direct LLM Answer: {answer_text}")
        
        if answer_text == "I don't know." or "I don't know" in answer_text:
            print(f"❌ LLM couldn't find relevant information")
            return {
                "answer": "No relevant information found in the PDF.",
                "source_documents": [],
                "has_rag_info": False,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "NONE",
                "selected_chunk_content": ""
            }
        else:
            # Return the enhanced content with media information
            has_rag_info = True
            source = [{
                "page_number": docs[0].metadata.get("pages", "N/A"),
                "content": answer_text  # Return the LLM's processed answer
            }]
            
            result = {
                "answer": answer_text,
                "source_documents": source,
                "has_rag_info": has_rag_info,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "PROCESSED",
                "selected_chunk_content": answer_text
            }
            return result

    except Exception as e:
        logger.error(f"Error in document search: {e}")
        return {
            "answer": "Error occurred during document search.",
            "source_documents": [],
            "has_rag_info": False,
            "dtc_code": dtc_code,
            "vehicle_info": vehicle_info,
            "selected_chunk_label": "ERROR",
            "selected_chunk_content": ""
        }

def extract_media_references_enhanced(content: str):
    """Extract image links and table references from content (same as rag_test_basic)"""
    media_info = {
        'images': [],
        'tables': [],
        'has_media': False
    }
    
    # Look for image references (common patterns)
    image_patterns = [
        r'!\[.*?\]\((.*?)\)',  # Markdown images
        r'<img.*?src=["\']([^"\']+)["\']',  # HTML images
        r'Image:\s*([^\s\n]+)',  # Custom image format
        r'Figure\s+\d+[:\-]?\s*([^\n]+)',  # Figure references
        r'\[Image:\s*([^\]]+)\]',  # Bracketed image refs
    ]
    
    for pattern in image_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        media_info['images'].extend(matches)
    
    # Look for table references
    table_patterns = [
        r'Table\s+\d+[:\-]?\s*([^\n]+)',  # Table references
        r'\|.*\|.*\|',  # Markdown table rows
        r'<table.*?</table>',  # HTML tables
    ]
    
    for pattern in table_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        media_info['tables'].extend(matches)
    
    media_info['has_media'] = bool(media_info['images'] or media_info['tables'])
    return media_info

def format_content_with_media_enhanced(content: str, doc_num: int):
    """Format content and extract media references (same as rag_test_basic)"""
    media_info = extract_media_references_enhanced(content)
    formatted_content = content
    
    # Add media information if found
    if media_info['has_media']:
        media_section = f"\n[MEDIA IN DOC {doc_num}]"
        
        if media_info['images']:
            media_section += f"\n📷 Images/Figures: {len(media_info['images'])} found"
            for i, img in enumerate(media_info['images'][:3], 1):  # Show first 3
                media_section += f"\n  - Image {i}: {img[:100]}{'...' if len(img) > 100 else ''}"
        
        if media_info['tables']:
            media_section += f"\n📊 Tables: {len(media_info['tables'])} found"
            for i, table in enumerate(media_info['tables'][:2], 1):  # Show first 2
                table_preview = table[:150].replace('\n', ' ')
                media_section += f"\n  - Table {i}: {table_preview}{'...' if len(table) > 150 else ''}"
        
        formatted_content += media_section
    
    return formatted_content

@tool
def grade_document_relevance(question: str, document_content: str, chunk_label: str = "UNKNOWN") -> dict:
    """Grade the relevance of retrieved documents (EXACT from version3_refactor)."""
    logger.info(f"📊 Grading relevance for {chunk_label}")
    
    print(f"🔍 GRADING DEBUG: chunk_label={chunk_label}, content_length={len(document_content if document_content else '')}")

    if not document_content or document_content == "No relevant information found in the PDF.":
        logger.info(f"📉 {chunk_label} has no content, score=0")
        result = {"relevance_score": 0, "graded": True, "chunk": chunk_label}
        print(f"🔍 GRADING RESULT: {result}")
        return result
    
    # Truncate for speed
    if len(document_content) > 300:
        truncated_content = document_content[:1000] + "..."
        logger.info(f"✂️ Truncated {chunk_label} from {len(document_content)} to 1000 chars")
    else:
        truncated_content = document_content
    
    try:
        score = retrieval_grader.invoke({"question": question, "documents": truncated_content})
        relevance_score = score.get('score', 0)
        logger.info(f"✅ {chunk_label} relevance score = {relevance_score}")
        result = {
            "relevance_score": relevance_score,
            "graded": True,
            "chunk": chunk_label
        }
        print(f"🔍 GRADING RESULT: {result}")
        return result
    except Exception as e:
        logger.error(f"⚠️ Error grading {chunk_label}: {e}")
        result = {"relevance_score": 0, "graded": True, "chunk": chunk_label}
        print(f"🔍 GRADING RESULT (error fallback): {result}")
        return result

@tool
def search_web_for_vehicle_info(query: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search web for additional vehicle diagnostic information using Tavily (EXACT from version3_refactor)."""
    logger.info("🌐 Performing Tavily web search")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.warning("No Tavily API key found, skipping web search")
        return {"results": [], "success": False, "error": "Missing TAVILY_API_KEY"}
    
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=tavily_api_key)
    except ImportError:
        logger.warning("Tavily package not installed, skipping web search")
        return {"results": [], "success": False, "error": "Tavily package not available"}
    
    search_term = dtc_code or vehicle_info or query
    
    if dtc_code:
        search_query = f"{search_term} vehicle diagnostic trouble code causes solutions"
        logger.info(f"Searching for DTC code: {search_term}")
    elif vehicle_info:
        search_query = f"{search_term} common problems solutions"
        logger.info(f"Searching for vehicle info: {search_term}")
    else:
        search_query = search_term
        logger.info(f"Searching for general query: {search_term}")
    
    try:
        response = tavily_client.search(query=search_query, max_results=3, search_depth="advanced")
        results = response.get('results', [])
        logger.info(f"Found {len(results)} results from web search")
        
        return {
            "results": results,
            "success": True,
            "query_used": search_query
        }
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return {"results": [], "success": False, "error": str(e)}

@tool
def search_youtube_videos(query: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search YouTube for diagnostic videos (EXACT from version3_refactor)."""
    logger.info("📺 Performing YouTube search")
    
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API key not found in environment variables")
        return {"youtube_results": [], "success": False,"error": "Missing YOUTUBE_API_KEY"}
    
    search_term = dtc_code or vehicle_info or query
    
    if dtc_code:
        search_query = f"{search_term} diagnostic trouble code repair"
        logger.info(f"Searching YouTube for DTC code: {search_term}")
    elif vehicle_info:
        search_query = f"{search_term} repair maintenance"
        logger.info(f"Searching YouTube for vehicle: {search_term}")
    else:
        search_query = f"car {search_term}"
        logger.info(f"Searching YouTube for general query: {search_term}")
    
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": search_query,
        "key": YOUTUBE_API_KEY,
        "maxResults": 4,
        "type": "video"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"YouTube API error: {response.status_code} {response.text}")
            return {"youtube_results": [], "success": False}
        
        data = response.json()
        videos = []
        
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            videos.append({
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id,
                "title": item["snippet"]["title"],
                "thumbnail_hq": item["snippet"]["thumbnails"].get("high", {}).get("url", ""),
                "thumbnail_max": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            })
        
        logger.info(f"Found {len(videos)} YouTube videos")
        return {
            "youtube_results": videos,
            "success": True,
            "query_used": search_query
        }
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return {"youtube_results": [], "success": False, "error": str(e)}

@tool
def format_diagnostic_results(
    question: str,
    rag_answer: str,
    web_results: Optional[List[dict]] = None,
    youtube_results: Optional[List[dict]] = None,
    dtc_code: Optional[str] = None,
    relevance_score: int = 0,
) -> dict:
    """Format the final diagnostic results with proper structure (EXACT from version3_refactor)."""
    logger.info("📝 Formatting final results")
    
    print(f"🔍 DEBUG: Received relevance_score = {relevance_score}")
    
    # Handle the rag_answer properly - it might be JSON string from tool result
    if isinstance(rag_answer, str):
        try:
            # Try to parse as JSON first
            rag_data = json.loads(rag_answer)
            rag_content = rag_data.get('answer', rag_answer)
            print(f"✅ Parsed JSON, extracted answer field")
        except json.JSONDecodeError:
            # If not JSON, use as-is
            rag_content = rag_answer
            print(f"✅ Using raw string content")
    else:
        rag_content = str(rag_answer)

    # Check if RAG found relevant information AND has good relevance score
    has_rag_info = (rag_content and 
                   rag_content != "No relevant information found in the PDF." and
                   "No relevant information found" not in rag_content)
    
    # CRITICAL: Use relevance score to decide formatting logic
    use_rag = has_rag_info and relevance_score == 1  # ← BOTH conditions needed
    
    print(f"🔍 DEBUG: has_rag_info={has_rag_info}, relevance_score={relevance_score}, use_rag={use_rag}")
    
    # If RAG has good info AND good relevance score, process it for steps/images/tables
    if use_rag:
        logger.info("Using RAG answer with step/image/table formatting")
        processed_rag_content = process_content_with_inline_images(rag_content)
        return {"formatted_response": processed_rag_content}
    
    # If RAG is not relevant (score 0) OR has no content, use web search
    if not use_rag and (web_results or youtube_results):
        logger.info("No RAG info found OR low relevance score, using web search formatting logic")
        print(f"🌐 Using web search logic - relevance_score={relevance_score}")
        
        # Combine web search content
        web_content = "\n\n".join([r.get("content", "") for r in web_results or [] if "content" in r])
        source_urls = [r.get("url", "") for r in web_results or [] if "url" in r][:3]
        youtube_links = [video.get("url", "") for video in youtube_results or []][:4]
        
        # Choose appropriate prompt based on presence of DTC code
        if dtc_code:
            prompt_template = f"""
You are an expert automotive diagnostic technician analyzing the Diagnostic Trouble Code (DTC): {dtc_code}

Based on the following information:
RAG ANSWER: {rag_answer}
WEB SEARCH RESULTS: {web_content}

Create a comprehensive diagnostic report that STRICTLY follows this EXACT format:

Category: [one-line description of what this DTC code represents]

Potential Causes:
- [cause 1]
- [cause 2]
- [continue until you have up to 3 causes, be specific and technical]

Diagnostic Steps:
- [step 1]
- [step 2]
- [continue until you have up to 3 clear diagnostic steps]

Possible Solutions:
- [solution 1]
- [solution 2]
- [continue until you have up to 3 solutions, be specific and technical]

Your response MUST follow this format exactly, with these exact section headings.
Be concise and technical in your bullet points. Do not add any other sections or explanations.
"""
        else:
            prompt_template = f"""
You are an automotive expert assistant. Create a comprehensive response based on the following information:

QUESTION: {question}

RAG ANSWER: {rag_answer}

WEB SEARCH RESULTS: 
{web_content}

Provide a detailed, helpful response that synthesizes all this information.
If the RAG answer indicates "No relevant information found in the PDF", prioritize the web search results.
"""
        
        try:
            response = llm.invoke(prompt_template)
            diagnostic_content = response.content.strip()
            
            # Build the final formatted answer
            final_answer = diagnostic_content
            
            # Add source URLs section (web)
            if source_urls:
                final_answer += "\n\n🕸️ Web Sources:\n"
                for url in source_urls:
                    final_answer += f"- {url}\n"
            
            # Add YouTube links section
            if youtube_links:
                final_answer += "\n\n📺 YouTube Diagnostic Videos:\n"
                for link in youtube_links:
                    final_answer += f"- {link}\n"
            
            logger.info("Generated structured diagnostic report from web sources")
            return {"formatted_response": final_answer}
        
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return {"formatted_response": rag_content}  # Fallback to original answer
    
    # Fallback: Return RAG content even if relevance is low (no web results available)
    logger.info("Fallback: Using RAG content despite low relevance (no web results)")
    print("⚠️ Fallback: No web results, using RAG content despite low relevance")
    processed_rag_content = process_content_with_inline_images(rag_content)
    return {"formatted_response": processed_rag_content}

def process_content_with_inline_images(content: str) -> str:
    """Process content to display images and tables inline with steps (EXACT from version3_refactor)."""
    
    if not content or len(content) < 10:
        return content
    
    try:
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check for numbered steps
            if re.match(r'^\d+\.\s+', line):
                processed_lines.append(f"**{line}**")
            
            # Check for YES/NO decision points and bullet points
            elif line.startswith(('YES -', 'NO -', '- YES', '- NO', '-')):
                processed_lines.append(f"   **{line}**")
            
            # Check for image references - show markdown as-is
            elif '![Image](' in line:
                processed_lines.append(f"   {line}")
                processed_lines.append("")  # Space after image
            
            # Check for table rows (contains | characters)
            elif '|' in line and len([c for c in line if c == '|']) >= 2:
                processed_lines.append(line)  # Keep table formatting as-is
            
            # Check for table separators
            elif '---' in line and '|' in line:
                processed_lines.append(line)
            
            # Check for questions (ending with ?)
            elif line.endswith('?'):
                processed_lines.append(f"\n**{line}**")
            
            else:
                if line:
                    processed_lines.append(line)
        
        return '\n'.join(processed_lines)
        
    except Exception as e:
        logger.error(f"Error in process_content_with_inline_images: {e}")
        return content

# Export all tools
__all__ = [
    "is_vehicle_related",
    "extract_vehicle_model", 
    "search_vehicle_documents",
    "enhance_question_with_vehicle",
    "grade_document_relevance",
    "search_web_for_vehicle_info",
    "search_youtube_videos",
    "format_diagnostic_results",
    "llm",
    "retriever",
]

# --- START: structured format_diagnostic_results (added by assistant) ---
@tool  
def format_diagnostic_results_structured(question: str, rag_answer: str, web_results: list = None, youtube_results: list = None, dtc_code: str = None, relevance_score: int = 0) -> dict:
    """Format diagnostic results with structured output for both voice and detailed report."""
    import re, urllib.parse
    logger.info("📝 Formatting structured diagnostic results")
    
    # Auto-extract DTC code from question if not provided
    if not dtc_code:
        dtc_match = re.search(r"\b([PBUC]\d{4})\b", question.upper())
        if dtc_match:
            dtc_code = dtc_match.group(1)
            logger.info(f"🔧 Auto-extracted DTC code: {dtc_code}")
    
    try:
        web_results = web_results or []
        youtube_results = youtube_results or []

        # Smart fallback: If no results provided, search for them
        if not web_results or not youtube_results:
            logger.info(f"🔍 Missing data, performing search for: {question}")
            
            if not web_results:
                try:
                    web_search_result = search_web_for_vehicle_info(question)
                    if web_search_result.get("success") and web_search_result.get("results"):
                        web_results = web_search_result["results"]
                        logger.info(f"🌐 Found {len(web_results)} web results via fallback")
                except Exception as e:
                    logger.warning(f"Failed to search web: {e}")

            if not youtube_results:
                try:
                    youtube_search_result = search_youtube_videos(question)
                    if youtube_search_result.get("success") and youtube_search_result.get("youtube_results"):
                        youtube_results = youtube_search_result["youtube_results"]
                        logger.info(f"📺 Found {len(youtube_results)} YouTube videos via fallback")
                except Exception as e:
                    logger.warning(f"Failed to search YouTube: {e}")

        # Create main diagnostic content
        final_answer = rag_answer or ""
        
        # Process web results into structured data
        source_urls = []
        if isinstance(web_results, list) and web_results:
            for r in web_results:
                if isinstance(r, dict):
                    url = r.get("url", "")
                    title = r.get("title", url)
                    snippet = r.get("content", "")[:200]
                    source_urls.append({"title": title, "url": url, "snippet": snippet})

        # Process YouTube results into structured data
        youtube_links_list = []
        if isinstance(youtube_results, list) and youtube_results:
            for y in youtube_results:
                if isinstance(y, dict):
                    url = y.get("url") or y.get("link") or ""
                    title = y.get("title") or url
                    vid = None
                    if "youtube.com" in url and "v=" in url:
                        vid = url.split("v=")[1].split("&")[0]
                    thumbnail = f"https://img.youtube.com/vi/{vid}/mqdefault.jpg" if vid else None
                    youtube_links_list.append({"title": title, "url": url, "video_id": vid, "thumbnail": thumbnail})

        # Create TTS summary
        tts_summary = ""
        try:
            if dtc_code:
                tts_prompt = f"Create a brief 3-sentence summary for voice about {dtc_code}. Simple language, no URLs."
            else:
                tts_prompt = f"Create a brief 3-sentence summary for voice about: {question}. Simple language, no URLs."
            
            tts_resp = llm.invoke(tts_prompt)
            tts_summary = tts_resp.content.strip() if hasattr(tts_resp, "content") else ""
            
            # Clean TTS summary
            if tts_summary:
                tts_summary = re.sub(r"https?://[^\s]+", "", tts_summary)
                tts_summary = re.sub(r"www\.[^\s]+", "", tts_summary)
                tts_summary = re.sub(r"[#*•\-`\[\]{}|\\]", "", tts_summary)
                tts_summary = re.sub(r"\s+", " ", tts_summary).strip()
                
        except Exception as e:
            logger.warning(f"Failed to generate TTS: {e}")
            tts_summary = f"Diagnostic information found for {dtc_code or 'your vehicle issue'}. Check the detailed report for solutions."

        payload = {
            "voice_output": tts_summary,
            "diagnostic_report": {
                "content": final_answer,
                "web_sources": source_urls,
                "youtube_videos": youtube_links_list
            },
            "dtc_code": dtc_code,
            "relevance_score": relevance_score
        }
        
        logger.info(f"📊 Created response: voice={len(tts_summary)} chars, content={len(final_answer)} chars, web={len(source_urls)}, youtube={len(youtube_links_list)}")
        
        # DUAL-CHANNEL SOLUTION:
        # Channel 1: Return only TTS-friendly voice output for chat/TTS
        # Channel 2: Store diagnostic data in a global variable that frontend can access
        
        diagnostic_data = {
            "content": final_answer,
            "web_sources": source_urls,
            "youtube_videos": youtube_links_list,
            "dtc_code": dtc_code,
            "relevance_score": relevance_score,
            "timestamp": json.dumps(datetime.now(), default=str) if 'datetime' in globals() else str(int(time.time() * 1000))
        }
        
        # Store diagnostic data globally (will be accessible via a separate endpoint)
        global _latest_diagnostic_data
        _latest_diagnostic_data = diagnostic_data
        
        logger.info(f"🔊 BACKEND: Returning voice output for TTS: {tts_summary}")
        logger.info(f"📊 BACKEND: Stored diagnostic data separately - content={len(final_answer)} chars, web_sources={len(source_urls)}, youtube_videos={len(youtube_links_list)}")
        
        return tts_summary
        
    except Exception as e:
        logger.error(f"Error in format_diagnostic_results_structured: {e}")
        fallback_tts = f"Diagnostic information found for {dtc_code or 'your vehicle issue'}. Check the detailed report for solutions."
        logger.info(f"🔊 BACKEND (fallback): Returning only voice output: {fallback_tts}")
        return fallback_tts

try:
    format_diagnostic_results = format_diagnostic_results_structured
except Exception:
    pass

def get_latest_diagnostic_data():
    """Return the latest diagnostic data stored globally"""
    global _latest_diagnostic_data
    return _latest_diagnostic_data

def clear_diagnostic_data():
    """Clear the stored diagnostic data"""
    global _latest_diagnostic_data
    _latest_diagnostic_data = None

# --- END: structured format_diagnostic_results ---
