import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from workflows.diagnostic_workflow import AsyncDiagnosticAgent


class TestWorkflowExecution:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workflow_success_path(self):
        agent = AsyncDiagnosticAgent()
        # Patch the underlying graph build to simulate tool invocation results
        with patch.object(agent, "_build_graph") as mock_build:
            mock_graph = AsyncMock()
            msg = MagicMock()
            msg.content = "Diagnostic result"
            msg.tool_calls = [{"name": "search_vehicle_documents"}]
            mock_graph.ainvoke.return_value = {"messages": [msg]}
            mock_build.return_value = mock_graph
            await agent.initialize()
            resp = await agent.chat("What is P0420?")
            assert "Diagnostic" in resp
            await agent.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        agent = AsyncDiagnosticAgent()
        with patch.object(agent, "_build_graph") as mock_build:
            mock_graph = AsyncMock()
            def mkmsg(i):
                m = MagicMock()
                m.content = f"Resp {i}"
                m.tool_calls = [{"name": "search_vehicle_documents"}]
                return m
            mock_graph.ainvoke.side_effect = lambda *a, **k: {"messages": [mkmsg(len(a)+len(k))]}
            mock_build.return_value = mock_graph
            await agent.initialize()
            tasks = [agent.chat(f"Q{i}") for i in range(5)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            await agent.cleanup()
