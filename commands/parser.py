"""
commands/parser.py — Command recognition via exact phrase matching.

Responsibilities:
- Accept a transcribed text string
- Normalize input (lowercase, strip punctuation) and compare against all
  entries in command_list.COMMANDS
- Support exact match as the primary strategy; optionally fuzzy-match
  as a fallback before escalating to the LLM
- Return the matched Command object (including response audio path and any
  associated hardware action) or None if no match found
- Be fast: this check must complete before deciding whether to call ChatGPT
"""
