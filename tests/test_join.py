"""test_join.py Unit testing of join.py"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from clozify_llm.constants import CLOZE_COL, DEFN_COL, WORD_COL
from clozify_llm.join import Joiner


@pytest.fixture
def sample_df_cloze() -> pd.DataFrame:
    """Fixture for unit testing"""
    return pd.DataFrame(
        {
            CLOZE_COL: ["cloze1", "cloze2"],
            f"{CLOZE_COL}_embedding": ["[0.1, 0.9, 0.1]", "[0.9, 0.1, 0.9]"],
        }
    )


@pytest.fixture
def sample_df_vocab() -> pd.DataFrame:
    """Fixture for unit testing"""
    return pd.DataFrame(
        {
            WORD_COL: ["word_a", "word_b"],
            DEFN_COL: ["define word_a", "define word_b"],
            f"{WORD_COL}_embedding": ["[1, 0, 1]", "[0, 1, 0]"],
        }
    )


@pytest.fixture
def sample_joiner(sample_df_cloze, sample_df_vocab) -> Joiner:
    """Joiner fixture for unit testing

    Uses embeddings where it is assumed cloze1 corresponds to word_b and cloze_2 corresponds to word_a
    """
    return Joiner(
        df_cloze=sample_df_cloze,
        df_vocab=sample_df_vocab,
    )


@pytest.fixture
def sample_join_result(sample_df_cloze, sample_df_vocab) -> pd.DataFrame:
    """Fixture describing result of join"""
    vocab_with_cloze_idx = sample_df_vocab.copy()
    vocab_with_cloze_idx["cloze_idx"] = [1, 0]
    vocab_with_cloze_idx["vocab_idx"] = [0, 1]
    df = pd.merge(sample_df_cloze, vocab_with_cloze_idx, left_index=True, right_on="cloze_idx")
    return df


def test_joiner_join_emb_sim(sample_joiner, sample_join_result):
    """Test behavior of Joiner.join_emb_sim()"""
    result = sample_joiner.join_emb_sim()

    expected = sample_join_result

    result_sorted = result.reindex(sorted(result.columns), axis=1).reset_index(drop=True)
    expected_sorted = expected.reindex(sorted(expected.columns), axis=1).reset_index(drop=True)
    assert_frame_equal(result_sorted, expected_sorted)


def test_joiner_clean_join_from_review(sample_joiner, sample_join_result):
    """Test behavior of Joiner.clean_join_from_review()"""
    candidate_join = sample_join_result
    manual_review = pd.DataFrame({"issue": [True, True], "cloze_idx": [0, 1], "correct_vocab_idx": ["None", 1]})
    result = sample_joiner.clean_join_from_review(candidate_join, manual_review, output_intermediate_cols=False)

    # Manual review stated cloze_idx 0 is not in vocab and cloze_idx 1 should correspond to vocab_idx 1
    # clean_join_from_view also does not carry through embedding cols
    expected = sample_join_result.query("cloze_idx==1")
    expected["vocab_idx"] = [1]
    expected[sample_joiner.word_col] = ["word_b"]
    expected[sample_joiner.defn_col] = ["define word_b"]
    expected = expected.drop(columns=[sample_joiner.word_emb_col, sample_joiner.cloze_emb_col])

    result_sorted = result.reindex(sorted(result.columns), axis=1).reset_index(drop=True)
    expected_sorted = expected.reindex(sorted(expected.columns), axis=1).reset_index(drop=True)
    assert_frame_equal(result_sorted, expected_sorted)


def test_joiner_view_multi_defn(sample_joiner):
    """Test behavior of Joiner.view_multi_defn()"""
    vocab = pd.DataFrame(
        {
            sample_joiner.word_col: ["my_word", "my_word", "my_unique_word"],
            sample_joiner.defn_col: ["defn 1", "defn 2", "unique defn"],
        }
    )
    result = sample_joiner.view_multi_defn(vocab)

    expected = vocab.loc[[0, 1]]

    assert_frame_equal(result, expected)
