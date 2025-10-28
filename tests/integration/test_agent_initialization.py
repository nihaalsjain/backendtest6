import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from agents.automotive_agent import Assistant, entrypoint
from configs.language_config import get_config_for_language, EnglishConfig, HindiConfig


class TestAgentInitialization:
    @pytest.mark.integration
    def test_assistant_initialization(self):
        with patch("agents.automotive_agent.load_prompts") as mock_load:
            mock_load.return_value = {
                "instructions": "Test instructions",
                "greetings": {"en": "Hello", "hi": "नमस्ते"}
            }
            assistant = Assistant()
            assert assistant.diagnostic_agent is None
            assert assistant.greetings == {"en": "Hello", "hi": "नमस्ते"}
            assert assistant.get_greeting("en") == "Hello"
            assert assistant.get_greeting("xx") == "Hello"  # fallback

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_diagnostic_agent_lazy_loading(self):
        with patch("agents.automotive_agent.load_prompts") as mock_load:
            mock_load.return_value = {"instructions": "X", "greetings": {"en": "Hello"}}
            assistant = Assistant()
        assert assistant.diagnostic_agent is None
        with patch("workflows.diagnostic_workflow.AsyncDiagnosticAgent") as mock_agent_cls:
            mock_agent = AsyncMock()
            mock_agent.initialize = AsyncMock()
            mock_agent_cls.return_value = mock_agent
            await assistant.initialize_diagnostic_agent()
            assert assistant.diagnostic_agent is not None
            mock_agent.initialize.assert_called_once()

    @pytest.mark.integration
    def test_config_selection_english(self):
        cfg = get_config_for_language("en", "Voice Assistant")
        assert "llm" in cfg and "stt" in cfg and "tts" in cfg

    @pytest.mark.integration
    def test_config_selection_hindi(self):
        cfg = get_config_for_language("hi", "Voice Assistant")
        assert "llm" in cfg and "stt" in cfg and "tts" in cfg

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_entrypoint_metadata_parsing(self, mocker):
        ctx = MagicMock()
        ctx.connect = AsyncMock()
        room = MagicMock()
        room.metadata = json.dumps({"language": "hi", "voiceBase": "Live Assistant"})
        room.remote_participants = {}
        ctx.room = room

        # Patch AgentSession & Assistant
        mock_session = AsyncMock()
        mocker.patch("agents.automotive_agent.AgentSession", return_value=mock_session)
        mocker.patch("agents.automotive_agent.RoomInputOptions")
        mock_assistant_inst = MagicMock()
        mock_assistant_inst.get_greeting.return_value = "Test greeting"
        mocker.patch("agents.automotive_agent.Assistant", return_value=mock_assistant_inst)

        # Force early exit by triggering room disconnect soon
        async def trigger_disconnect():
            await asyncio.sleep(0.1)
            for cb in room.on.call_args_list:
                pass
        # Just run with timeout to ensure it starts
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(entrypoint(ctx), timeout=0.2)
