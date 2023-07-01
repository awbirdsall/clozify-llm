"""conftest.py Shared test fixtures
"""

import pytest
from openai.openai_object import OpenAIObject


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


@pytest.fixture
def file_create_response():
    """Mock response from File API"""
    return OpenAIObject.construct_from(
        {
            "id": "file-XjGxS3KTG0uNmNOK362iJua3",
            "object": "file",
            "bytes": 140,
            "created_at": 1613779121,
            "filename": "mydata.jsonl",
            "purpose": "fine-tune",
        }
    )


@pytest.fixture
def finetune_create_response(embedding_vals):
    """Mock response from Finetune API"""
    return OpenAIObject.construct_from(
        {
            "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
            "object": "fine-tune",
            "model": "curie",
            "created_at": 1614807352,
            "events": [
                {
                    "object": "fine-tune-event",
                    "created_at": 1614807352,
                    "level": "info",
                    "message": "Job enqueued. Waiting for jobs ahead to complete. Queue number: 0.",
                }
            ],
            "fine_tuned_model": None,
            "hyperparams": {
                "batch_size": 4,
                "learning_rate_multiplier": 0.1,
                "n_epochs": 4,
                "prompt_loss_weight": 0.1,
            },
            "organization_id": "org-...",
            "result_files": [],
            "status": "pending",
            "validation_files": [],
            "training_files": [
                {
                    "id": "file-XGinujblHPwGLSztz8cPS8XY",
                    "object": "file",
                    "bytes": 1547276,
                    "created_at": 1610062281,
                    "filename": "my-data-train.jsonl",
                    "purpose": "fine-tune-train",
                }
            ],
            "updated_at": 1614807352,
        }
    )
