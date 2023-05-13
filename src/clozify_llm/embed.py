"""embed.py Generate embeddings from embedding model
"""
from typing import Optional

import pandas as pd

from clozify_llm.constants import CLOZE_COL, WORD_COL
from clozify_llm.utils import get_embs


def add_emb(df: pd.DataFrame, input_col: Optional[str] = None) -> pd.DataFrame:
    """Return copy of input dataframe with new embedding column

    Parameters
    ----------
    df : pd.DataFrame
      DataFrame containing str column to embed.
    input_col : str, optional, default None
      If provided, column in df to get embeddings for. Otherwise use WORD_COL or CLOZE_COL.

    Note this calls the embedding API len(df) times.
    """
    if input_col is not None:
        to_embed = input_col
    elif WORD_COL in df.columns:
        to_embed = WORD_COL
    elif CLOZE_COL in df.columns:
        to_embed = CLOZE_COL
    else:
        raise ValueError(f"input_col must be set or input dataframe must contain {WORD_COL} or {CLOZE_COL}")
    output_col = f"{to_embed}_embedding"
    df_out = df.copy()
    df_out[output_col] = get_embs(df_out[to_embed].tolist())
    return df_out
