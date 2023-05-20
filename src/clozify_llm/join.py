"""join.py Join existing cloze and vocab for training
"""
from ast import literal_eval
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from clozify_llm.constants import CLOZE_COL, DEFN_COL, WORD_COL


class Joiner:
    """Joins existing cloze and vocab for training

    Assumes both already have embedding columns
    """

    def __init__(
        self,
        df_cloze: pd.DataFrame,
        df_vocab: pd.DataFrame,
        cloze_col: str = CLOZE_COL,
        word_col: str = WORD_COL,
        defn_col: str = DEFN_COL,
        cloze_emb_col: Optional[str] = None,
        word_emb_col: Optional[str] = None,
    ):
        self.df_cloze = df_cloze
        self.df_vocab = df_vocab
        self.cloze_col = cloze_col
        self.word_col = word_col
        self.defn_col = defn_col
        if cloze_emb_col is None:
            cloze_emb_col = f"{cloze_col}_embedding"
        if word_emb_col is None:
            word_emb_col = f"{word_col}_embedding"
        self.cloze_emb_col = cloze_emb_col
        self.word_emb_col = word_emb_col

    def join_emb_sim(self) -> pd.DataFrame:
        """Join df_cloze and df_vocab based on cosine similarity of embedding
        columns

        1-to-many join where each cloze sentence appears once but each row in  vocab
        could occur 0, 1, or multiple times.

        Note the function return should still be manually reviewed.

        This is an imprecise match due to following complications:
        - Clozes can include vocabulary words in non-standard form (inflected,
        conjugated)
        - Vocabulary is occassionally phrases or hyphenated words for which the
        cloze picks out just a single word.
        - Vocabulary occassionally includes duplicate word with different meanings.
        Cloze should correspond not just to correct vocab word but also correct
        definition.

        Also note each vocab generated zero, one, or more cloze sentences. Each
        cloze sentence corresponds to exactly one vocab (word + definition).

        Returns
        -------
        pd.DataFrame
          DataFrame with proposed join between cloze and vocab
        """
        if self.cloze_emb_col not in self.df_cloze.columns:
            raise ValueError(f"cloze_emb_col {self.cloze_emb_col} must be in self.df_cloze")
        if self.word_emb_col not in self.df_vocab.columns:
            raise ValueError(f"word_emb_col {self.word_emb_col} must be in self.df_vocab")

        # Handle str of list (common if embedding dataframes loaded from csv)
        cloze_embs = self.df_cloze[self.cloze_emb_col]
        if cloze_embs.dtype == object:
            cloze_embs = cloze_embs.apply(lambda x: literal_eval(x))
        word_embs = self.df_vocab[self.word_emb_col]
        if word_embs.dtype == object:
            word_embs = word_embs.apply(lambda x: literal_eval(x))

        X = np.array(cloze_embs.tolist())
        Y = np.array(word_embs.tolist())
        cos_sim_word = cosine_similarity(X, Y)
        # For each cloze, identify the index of the word with closest embedding
        match_idx_word = np.argmax(cos_sim_word, axis=1)
        # Use the matches indices to create a join key
        join_keys = pd.Series(match_idx_word).reset_index()
        join_keys.columns = ["cloze_idx", "vocab_idx"]
        # Assign join keys to each vocab -- note multiple cloze_idx can map to
        # single vocab_idx
        vocab_with_keys = pd.merge(
            join_keys,
            self.df_vocab,
            left_on="vocab_idx",
            right_index=True,
            how="left",
            validate="m:1",
        )
        # join cloze and vocab using cloze_idx now in df_vocab.
        # 1:m join because each self.df_cloze should have unique cloze_idx
        # while each df_vocab could appear multiple times
        candidate_join = pd.merge(
            self.df_cloze,
            vocab_with_keys,
            left_index=True,
            right_on="cloze_idx",
            how="left",
            validate="1:m",
        )
        return candidate_join

    def clean_join_from_review(
        self, candidate_join: pd.DataFrame, manual_review: pd.DataFrame, output_intermediate_cols: bool = False
    ) -> pd.DataFrame:
        """Use the results from manual review to clean the candidate join

        The manual review should identify both (1) incorrect words in the candidate join and (2) incorrect definitions

        Parameters
        ----------
        candidate_join : pd.DataFrame
          Proposed join.
        manual_review : pd.DataFrame
          Manual corrections, with columns "issue", "cloze_idx", "correct_vocab_idx".
        """
        has_issue = manual_review[manual_review["issue"] == True]  # noqa: E712
        to_correct = has_issue.replace("None", None).dropna()
        to_correct["correct_vocab_idx"] = to_correct["correct_vocab_idx"].astype(int)
        correction = pd.merge(
            to_correct, self.df_vocab, left_on="correct_vocab_idx", right_index=True, how="left"
        ).drop(columns=["issue"])
        with_corrections = pd.merge(
            candidate_join.drop([self.cloze_emb_col, self.word_emb_col], axis="columns"),
            correction,
            left_on="cloze_idx",
            right_on="cloze_idx",
            how="left",
            suffixes=("", "_corrected"),
        )
        corrected_word_col = f"{self.word_col}_corrected"
        corrected_defn_col = f"{self.defn_col}_corrected"
        # Make replacements
        rows_to_correct = with_corrections[corrected_word_col].notnull()
        with_corrections.loc[rows_to_correct, "vocab_idx"] = with_corrections.loc[rows_to_correct, "translation"]
        with_corrections.loc[rows_to_correct, self.word_col] = with_corrections.loc[rows_to_correct, corrected_word_col]
        with_corrections.loc[rows_to_correct, self.defn_col] = with_corrections.loc[rows_to_correct, corrected_defn_col]
        # Drop rows not in vocab
        to_drop = has_issue.loc[has_issue["correct_vocab_idx"] == "None", "cloze_idx"].values
        with_corrections = with_corrections[~with_corrections.cloze_idx.isin(to_drop)]

        if not output_intermediate_cols:
            with_corrections = with_corrections.drop(
                columns=["correct_vocab_idx", corrected_word_col, corrected_defn_col, self.word_emb_col]
            )

        return with_corrections

    def view_multi_defn(self, vocab: pd.DataFrame) -> pd.DataFrame:
        """Helper function to view subset of vocab when single word can have multiple definitions"""
        return vocab[vocab.duplicated(subset=[self.word_col], keep=False)].sort_values(self.word_col)[
            [self.word_col, self.defn_col]
        ]
