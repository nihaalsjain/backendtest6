# setup.py
import sys
import os
import logging

os.environ["HF_HUB_ENABLE_SYMLINKS"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import json
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.RAG_config import PDF_FOLDER, MARKDOWN_DIR, CHUNKS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode

import chromadb
from chromadb.utils import embedding_functions
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    """Make sure all output directories exist before running pipeline."""
    for d in [MARKDOWN_DIR, CHUNKS_DIR, CHROMA_DB_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Ensured directories: {MARKDOWN_DIR}, {CHUNKS_DIR}, {CHROMA_DB_DIR}")


# -----------------------------
# 1️⃣ PDF Parsing with Docling
# -----------------------------
def convert_with_image_annotation(input_doc_path):
    pipeline_options = PdfPipelineOptions(
    do_ocr=True,                      # OCR text recognition if PDF is scanned
    do_table_structure=True,           # Detect and preserve table structure
    preserve_font_styles=True,         # ✅ Keep font sizes, bold/italic → helps with headings
    preserve_font_colors=True,         # Optional: keeps font color info
    preserve_layout=True,              # ✅ Preserve text layout (paragraphs, indentation)
    generate_page_images=True,        # Only needed if you want images
    enable_remote_services=False,       # For OCR / AI-powered enhancements
    do_picture_description=False,      # Only if you need image descriptions
)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    conv_res = converter.convert(source=input_doc_path)
    return conv_res

def export_single_md_with_images_and_serials(conv_res, output_path: Path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem.replace(" ", "_")
    md_filename = output_path / f"{doc_filename}-full-with-serials.md"

    conv_res.document.save_as_markdown(
        md_filename,
        image_mode=ImageRefMode.REFERENCED,
        include_annotations=True,
        page_break_placeholder="<!-- PAGE_BREAK -->",
    )

    # Add page-end markers
    with open(md_filename, "r", encoding="utf-8") as f:
        md_text = f.read()

    pages = md_text.split("<!-- PAGE_BREAK -->")
    final_md = ""
    for idx, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        final_md += page_text + f"\n\n<!-- PAGE {idx} END -->\n\n"

    Path(md_filename).write_text(final_md, encoding="utf-8")
    print(f"✅ Markdown saved with serials → {md_filename}")
    return md_filename

def parse_pdfs():
    print("📄 Parsing PDFs with Docling pipeline...")
    markdown_files = []
    pdf_dir = Path(PDF_FOLDER)  # ✅ convert str → Path
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"   ➡ Converting {pdf_path}")
        conv_res = convert_with_image_annotation(pdf_path)
        md_file = export_single_md_with_images_and_serials(conv_res, MARKDOWN_DIR)
        markdown_files.append(md_file)
    return markdown_files

# -----------------------------
# 2️⃣ Hybrid Chunking
# -----------------------------

def chunk_markdowns(markdown_files, min_chars=1000):
    
    all_chunks = []
    page_marker_pattern = re.compile(r"<!-- PAGE (\d+) END -->")
    
    # Convert CHUNKS_DIR to Path object
    chunks_dir = Path(CHUNKS_DIR)

    for md_file in markdown_files:
        md_file = Path(md_file)
        with md_file.open("r", encoding="utf-8") as f:
            md_text = f.read()

        # Split by headings (## or ###)
        sections = re.split(r"(##+ .+)", md_text)

        raw_chunks = []
        current_heading = ""
        current_content = ""
        current_pages = set()

        for sec in sections:
            if not sec.strip():
                continue

            heading_match = re.match(r"(##+)\s+(.+)", sec)
            if heading_match:
                # Save previous raw chunk
                if current_heading:
                    raw_chunks.append({
                        "heading": current_heading,
                        "pages": sorted(current_pages) if current_pages else [1],
                        "chunk_text": current_content.strip()
                    })
                # Start new raw chunk
                current_heading = heading_match.group(2).strip()
                current_content = sec
                current_pages = set(int(p) for p in page_marker_pattern.findall(sec))
            else:
                current_content += sec
                pages_in_sec = page_marker_pattern.findall(sec)
                current_pages.update(int(p) for p in pages_in_sec)

        # Add last raw chunk
        if current_heading:
            raw_chunks.append({
                "heading": current_heading,
                "pages": sorted(current_pages) if current_pages else [1],
                "chunk_text": current_content.strip()
            })

        # Combine small chunks
        combined_chunks = []
        buffer_chunk = {"heading": "", "pages": [], "chunk_text": ""}
        for chunk in raw_chunks:
            # If buffer is empty, start new buffer
            if not buffer_chunk["chunk_text"]:
                buffer_chunk = chunk.copy()
            else:
                # Combine if total length < min_chars
                if len(buffer_chunk["chunk_text"]) < min_chars:
                    # Merge headings and pages
                    buffer_chunk["chunk_text"] += "\n\n" + chunk["chunk_text"]
                    buffer_chunk["pages"] = sorted(set(buffer_chunk["pages"] + chunk["pages"]))
                else:
                    # Save buffer and start new
                    combined_chunks.append(buffer_chunk)
                    buffer_chunk = chunk.copy()

        # Add final buffer
        if buffer_chunk["chunk_text"]:
            combined_chunks.append(buffer_chunk)

        # Save JSON - Fixed line
        json_path = chunks_dir / f"{md_file.stem}_chunks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_chunks, f, ensure_ascii=False, indent=4)

        print(f"✅ {len(combined_chunks)} chunks saved → {json_path}")
        all_chunks.extend(combined_chunks)

    return all_chunks

# -----------------------------
# 3️⃣ Build Embeddings + Chroma
# -----------------------------

def build_embeddings(chunks):

    logger.info("🔎 Building embeddings with OpenAIEmbeddings + Chroma...")

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    collection_name = "pdf_chunks"
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        logger.info(f"🗑 Deleted existing collection '{collection_name}'")

    # Create new collection with OpenAI embedding function
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
    )

    ids = [f"chunk_{i+1}" for i in range(len(chunks))]
    texts = [c["chunk_text"] for c in chunks]
    metadatas = [{"pages": ",".join(map(str, c.get("pages", []))), "heading": c.get("heading", "")} for c in chunks]

    # Compute embeddings with OpenAIEmbeddings (debug print)
    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
    embeddings = embeddings_model.embed_documents(texts)
    for i, emb in enumerate(embeddings):
        logger.debug(f"Chunk {i+1} embedding length: {len(emb)}")

    # Add documents to Chroma
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    logger.info(f"✅ Stored {len(chunks)} chunks in ChromaDB @ {CHROMA_DB_DIR}")
    logger.info(f"📚 Collection '{collection_name}' now contains {len(collection.get()['ids'])} documents")

    return collection


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    ensure_dirs()
    markdowns = parse_pdfs()
    chunks = chunk_markdowns(markdowns)
    build_embeddings(chunks)
    logger.info("\n🎉 Setup complete! Multimodal RAG is ready.")

if __name__ == "__main__":
    main()