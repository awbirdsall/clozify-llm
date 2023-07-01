"""test_finetune.py Unit testing of finetune.py"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from clozify_llm.constants import CLOZE_COL, DEFN_COL, WORD_COL
from clozify_llm.finetune import FineTuner
from clozify_llm.utils import format_completion, format_prompt


@pytest.fixture
def training_file_path(tmp_path) -> str:
    return str(tmp_path / "training.jsonl")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample dataframe used as input to FineTuner"""
    df = pd.DataFrame(
        {
            WORD_COL: ["word"],
            DEFN_COL: ["defn"],
            "text": ["use my word"],
            "translation": ["translation with word"],
            CLOZE_COL: ["cloze"],
        }
    )
    return df


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Sample dataset as expected to be created by sample finetuner"""
    sample_prompt = format_prompt("word", "defn")
    sample_completion = format_completion("use my word", "translation with word", "cloze")
    return [
        {
            "prompt": sample_prompt,
            "completion": sample_completion,
        }
    ]


@pytest.fixture
def sample_finetuner(sample_df, training_file_path) -> FineTuner:
    return FineTuner(
        df=sample_df,
        training_data_path=training_file_path,
        model="dummy_completion_model",
    )


class TestFinetune:
    def test_finetune_create_dataset(self, sample_finetuner, sample_dataset):
        result = sample_finetuner.create_dataset()

        expected = sample_dataset

        assert result == expected

    def test_finetune_write_data(self, sample_finetuner, sample_dataset, tmp_path):
        assert not Path(sample_finetuner.training_data_path).exists()

        sample_finetuner.write_data(sample_dataset, overwrite=False)

        assert Path(sample_finetuner.training_data_path).exists()

    @patch("clozify_llm.finetune.openai.FineTune")
    @patch("clozify_llm.finetune.openai.File")
    def test_finetune_start_finetuning(
        self, mock_file, mock_finetune, sample_finetuner, file_create_response, finetune_create_response
    ):
        mock_file.create.return_value = file_create_response
        mock_finetune.create.return_value = finetune_create_response

        assert not Path(sample_finetuner.training_data_path).exists()

        result = sample_finetuner.start_finetuning(generate_dataset=True)

        expected = finetune_create_response

        assert result == expected
