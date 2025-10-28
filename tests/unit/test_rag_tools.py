import pytest
from tools import RAG_tools

def test_set_current_language_valid():
    RAG_tools.set_current_language("hi")
    assert RAG_tools.CURRENT_LANGUAGE == "hi"


def test_set_current_language_invalid():
    prev = RAG_tools.CURRENT_LANGUAGE
    RAG_tools.set_current_language("xx")
    assert RAG_tools.CURRENT_LANGUAGE == prev
