import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from workflows.diagnostic_workflow import AsyncDiagnosticAgent
from configs.language_config import EnglishConfig, HindiConfig


class TestE2EConversation:
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_basic_conversation(self):
        agent = AsyncDiagnosticAgent()
        with patch.object(agent, "_build_graph") as mock_build:
            mock_graph = AsyncMock()
            msgs = []
            for txt in ["Hi there", "Here is the answer"]:
                m = MagicMock()
                m.content = txt
                m.tool_calls = [{"name": "search_vehicle_documents"}]
                msgs.append(m)
            mock_graph.ainvoke.return_value = {"messages": msgs}
            mock_build.return_value = mock_graph
            await agent.initialize()
            r1 = await agent.chat("Hello")
            r2 = await agent.chat("Question about engine")
            assert isinstance(r1, str) and isinstance(r2, str)
            await agent.cleanup()

    @pytest.mark.e2e
    def test_language_configs(self):
        en_cfg = EnglishConfig.get_config("Voice Assistant")
        hi_cfg = HindiConfig.get_config("Voice Assistant")
        for cfg in (en_cfg, hi_cfg):
            assert "llm" in cfg and "vad" in cfg
