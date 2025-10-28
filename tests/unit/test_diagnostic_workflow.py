import pytest
import asyncio
from workflows.diagnostic_workflow import AsyncDiagnosticAgent

@pytest.mark.asyncio
async def test_agent_initial_language_sync():
    agent = AsyncDiagnosticAgent(language="hi")
    await agent.initialize()
    # Internal language should match
    assert agent._language == "hi"
