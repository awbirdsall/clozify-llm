"""test_embed.py Testing of embed.py
"""
from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from clozify_llm.constants import CLOZE_COL, WORD_COL
from clozify_llm.embed import add_emb


@pytest.mark.parametrize(
    "df_col,input_col",
    (
        (CLOZE_COL, None),
        (WORD_COL, None),
        ("custom_input", "custom_input"),
    ),
)
@patch("clozify_llm.embed.get_embs")
def test_add_emb(mock_get_embs, df_col, input_col, embedding_vals):
    """Test behavior of add_emb with different input column names"""
    mock_get_embs.return_value = [embedding_vals]
    df = pd.DataFrame({df_col: ["Input str"]})
    result = add_emb(df, input_col=input_col)
    expected = df.copy()
    embedding_col = pd.Series(data=[embedding_vals])
    expected[f"{df_col}_embedding"] = embedding_col

    assert_frame_equal(result, expected)


@patch("clozify_llm.embed.get_embs")
def test_add_emb_input_not_set(mock_get_embs, embedding_vals):
    """Test behavior of add_emb when input_col is not set and no default column is present"""
    input_col = "input_col"
    mock_get_embs.return_value = [embedding_vals]
    df = pd.DataFrame({input_col: ["Input str"]})
    with pytest.raises(ValueError):
        add_emb(df)
