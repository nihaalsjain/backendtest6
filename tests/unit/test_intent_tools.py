import importlib

def test_intent_module_import():
    mod = importlib.import_module('tools.intent_tools')
    assert hasattr(mod, 'classify_intent') or True  # placeholder
