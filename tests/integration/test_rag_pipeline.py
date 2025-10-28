import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts import rag_setup


class TestRAGPipeline:
    @pytest.mark.integration
    def test_markdown_chunking_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Heading\nContent line 1.\n\n## Subhead\nMore content.\n")
            md_path = f.name
        try:
            with tempfile.TemporaryDirectory() as chunks_dir:
                with patch("scripts.rag_setup.CHUNKS_DIR", chunks_dir):
                    chunks = rag_setup.chunk_markdowns([md_path], max_tokens=200)
                    assert len(chunks) > 0
                    assert all("chunk_text" in c for c in chunks)
        finally:
            Path(md_path).unlink(missing_ok=True)

    @pytest.mark.integration
    @pytest.mark.requires_api
    def test_build_embeddings_mocked(self, mocker):
        chunks = [
            {"chunk_text": "P0420 catalyst efficiency", "pages": [1], "heading": "DTC"},
            {"chunk_text": "P0171 lean condition", "pages": [2], "heading": "Fuel"},
        ]
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.list_collections.return_value = []
        mock_client.create_collection.return_value = mock_collection
        mocker.patch("scripts.rag_setup.chromadb.PersistentClient", return_value=mock_client)
        mock_embeddings = MagicMock()
        mocker.patch("scripts.rag_setup.OpenAIEmbeddings", return_value=mock_embeddings)
        mock_embeddings.embed_documents.return_value = [[0.1] * 10, [0.2] * 10]
        rag_setup.build_embeddings(chunks)
        mock_collection.add.assert_called()
