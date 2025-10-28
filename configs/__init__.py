"""
Configs package.

Contains language-specific configuration builders for different assistants.

Modules:
    - language_config.py: Config for English assistants (OpenAI GPT or Gemini)
       Config for Hindi assistants (Sarvam/OpenAI or Gemini)
       Config for Kannada assistants (Sarvam/OpenAI or Gemini)

Each module exports:
    get_config_for_language(lang_code, voice_base) -> dict
"""
