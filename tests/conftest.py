"""conftest.py Shared test fixtures
"""

import pytest


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
