import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from clozify_llm.extract.extract_cloze import extract_cloze


@pytest.fixture
def raw_input():
    return [
        {
            "collectionClozeSentences": [
                {
                    "text": "Le {{chapeau}} est rouge.",
                    "translation": "The hat is red.",
                },
                {
                    "text": "Le {{foulard}} est bleu.",
                    "translation": "The scarf is blue.",
                },
            ],
            "collection": {"name": "Colors"},
        }
    ]


def test_extract_cloze(raw_input):
    expected_output = pd.DataFrame(
        {
            "text": ["Le {{chapeau}} est rouge.", "Le {{foulard}} est bleu."],
            "translation": ["The hat is red.", "The scarf is blue."],
            "collection": ["Colors", "Colors"],
            "cloze": ["chapeau", "foulard"],
        }
    )

    df = extract_cloze(raw_input)

    assert_frame_equal(df, expected_output)
