"""These tests cover functions for data processing: JSONL parsing, validating message structure, Dataset module checks"""

import json
import tempfile
from pathlib import Path

import pytest


#test parsing of jsonl for getting questions used to build openai prompts
def test_get_questions_parses_jsonl():
    from data.prompts import get_questions

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        data = {"question": "Test", "starter_code": "def solve():"}
        f.write(json.dumps(data) + "\n")
        temp_path = f.name

    try:
        questions = get_questions(temp_path)
        assert len(questions) == 1
        assert questions[0]["question"] == "Test"
    finally:
        Path(temp_path).unlink()

#test that message structure for openai prompt is correct
def test_build_message_structure():
    from data.prompts import build_message

    msg = build_message("What is 2+2?", "def answer():")

    assert msg[0]["role"] == "system"
    assert msg[1]["role"] == "user"
    assert "What is 2+2?" in msg[1]["content"]

#test that dataset module xists
def test_dataset_module():
    try:
        import model.dataset
        assert hasattr(model.dataset, 'CodeDataset')
    except ModuleNotFoundError:
        #skips if not installed
        pytest.skip("Required dependencies not installed")

