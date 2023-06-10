"""test_utils.py Unit testing of utils
"""
from unittest.mock import patch

import pytest

from clozify_llm.constants import END_STR, PROMPT_SEPARATOR
from clozify_llm.utils import format_completion, format_prompt, get_emb, get_embs


@pytest.fixture
def embedding_vals():
    """Mock values for embedding"""
    return [
        0.0023064255,
        -0.009327292,
        -0.0028842222,
    ]


@pytest.fixture
def embedding_response(embedding_vals):
    """Mock response from Embedding API"""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding_vals,
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


def test_format_completion():
    text = "text"
    translation = 'translation,"quotation",commas'
    cloze = "cloze"
    result = format_completion(text, translation, cloze)

    expected = ' "text","translation,\\"quotation\\",commas","cloze"' + END_STR

    assert result == expected


def test_format_prompt():
    word = " Word"
    definition = "This is a definition "
    result = format_prompt(word, definition)

    expected = f"Word\nThis is a definition{PROMPT_SEPARATOR}"

    assert result == expected


@patch("clozify_llm.utils.openai.Embedding")
def test_get_emb(mock_completion, embedding_response):
    mock_completion.create.return_value = embedding_response
    result = get_emb("Input str")
    expected = embedding_response

    assert result == expected


@patch("clozify_llm.utils.openai.Embedding")
def test_get_embs(mock_completion, embedding_response, embedding_vals):
    mock_completion.create.return_value = embedding_response
    result = get_embs(["Input str"])
    expected = [embedding_vals]
    assert result == expected
