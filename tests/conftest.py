"""
Shared fixtures and configuration for all tests.
"""

import os
import sys
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import pytest
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before importing modules
os.environ["TESTING"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LIVEKIT_URL", "wss://test.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "test-api-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-api-secret")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("YOUTUBE_API_KEY", "test-youtube-key")

# ============================================
# Event Loop Configuration
# ============================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for tests (Windows compatibility)."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.get_event_loop_policy()


@pytest.fixture
def event_loop(event_loop_policy):  # type: ignore
    """Create an event loop for async tests and ensure cleanup."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()

# ============================================
# Mock Fixtures for External Dependencies
# ============================================

@pytest.fixture
def mock_openai():
    """Mock OpenAI API calls used indirectly by langchain wrappers."""
    with patch("openai.OpenAI") as mock:
        client = MagicMock()
        mock.return_value = client

        # Mock chat completions
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="Test response"))]
        client.chat.completions.create.return_value = completion

        # Mock embeddings
        embedding = MagicMock()
        embedding.data = [MagicMock(embedding=[0.1] * 1536)]
        client.embeddings.create.return_value = embedding

        # Mock whisper/transcription
        transcription = MagicMock()
        transcription.segments = [
            {"start": 0, "end": 5, "text": "Test transcription"}
        ]
        client.audio.transcriptions.create.return_value = transcription

        yield client


@pytest.fixture
def mock_livekit():
    """Mock LiveKit core classes."""
    with patch("livekit.agents.AgentSession") as mock_session:
        with patch("livekit.agents.JobContext") as mock_context:
            session = AsyncMock()
            context = MagicMock()
            room = MagicMock()
            room.metadata = json.dumps({"language": "en", "voiceBase": "Voice Assistant"})
            room.remote_participants = {}
            context.room = room

            mock_session.return_value = session
            mock_context.return_value = context
            yield {"session": session, "context": context, "room": room}


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB vector store interactions."""
    with patch("chromadb.PersistentClient") as mock_client:
        client = MagicMock()
        collection = MagicMock()
        collection.add.return_value = None
        collection.query.return_value = {
            "ids": [["chunk_1", "chunk_2"]],
            "documents": [["Test document 1", "Test document 2"]],
            "metadatas": [[{"pages": "1"}, {"pages": "2"}]],
            "distances": [[0.1, 0.2]]
        }
        collection.get.return_value = {"ids": ["chunk_1"], "documents": ["Test document"]}
        client.create_collection.return_value = collection
        client.get_or_create_collection.return_value = collection
        client.list_collections.return_value = []
        mock_client.return_value = client
        yield client

# ============================================
# Test Data Fixtures
# ============================================

@pytest.fixture
def sample_dtc_question():
    return "I have error code P0420 on my 2019 Ford F-150"


@pytest.fixture
def sample_rag_document():
    return {
        "chunk_text": "P0420 indicates catalyst system efficiency below threshold",
        "pages": [1, 2],
        "heading": "Diagnostic Trouble Codes"
    }


@pytest.fixture
def sample_chat_context():
    from livekit.agents.llm import ChatContext, ChatMessage
    ctx = ChatContext()
    # Provide minimal structure expected by adapter
    ctx.items = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi! I'm Allion, your automotive assistant."),
        ChatMessage(role="user", content="I have code P0420")
    ]
    return ctx


@pytest.fixture
def temp_pdf_file():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n")
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_image_file():
    import base64
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_data)
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)

# ============================================
# Agent and Workflow Fixtures
# ============================================

@pytest.fixture
async def mock_diagnostic_agent():
    from workflows.diagnostic_workflow import AsyncDiagnosticAgent
    with patch.object(AsyncDiagnosticAgent, "_build_graph") as mock_build:
        mock_graph = AsyncMock()
        # Provide a message that looks like a proper tool-assisted response
        tool_msg = MagicMock()
        tool_msg.content = "P0420 indicates catalyst efficiency issue"
        tool_msg.tool_calls = [{"name": "search_vehicle_documents"}]
        mock_graph.ainvoke.return_value = {"messages": [tool_msg]}
        mock_build.return_value = mock_graph
        agent = AsyncDiagnosticAgent()
        await agent.initialize()
        yield agent
        await agent.cleanup()


@pytest.fixture
def mock_assistant():
    from agents.automotive_agent import Assistant
    with patch("agents.automotive_agent.load_prompts") as mock_load:
        mock_load.return_value = {
            "instructions": "Test instructions",
            "greetings": {"en": "Hello", "hi": "नमस्ते", "kn": "ನಮಸ್ಕಾರ"}
        }
        assistant = Assistant()
        yield assistant

# ============================================
# Utility Fixtures
# ============================================

@pytest.fixture
def assert_async_raises():
    async def _assert(exc_type, coro):
        with pytest.raises(exc_type):
            await coro
    return _assert


@pytest.fixture
def capture_logs():
    import logging
    from io import StringIO

    class LogCapture:
        def __init__(self):
            self.stream = StringIO()
            self.handler = logging.StreamHandler(self.stream)
            self.handler.setLevel(logging.DEBUG)
        def __enter__(self):
            logging.getLogger().addHandler(self.handler)
            return self
        def __exit__(self, *args):
            logging.getLogger().removeHandler(self.handler)
        def get_logs(self):
            return self.stream.getvalue()
    return LogCapture()

# ============================================
# Environment Management
# ============================================

@pytest.fixture(autouse=True)
def reset_environment():
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_env_vars():
    def _set_vars(**kwargs: Any):
        for k, v in kwargs.items():
            os.environ[k] = v
    return _set_vars
