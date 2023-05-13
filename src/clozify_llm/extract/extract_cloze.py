"""extract-cloze.py Utility for extracting tabular cloze data from raw json
"""
import pandas as pd


def extract_cloze(raw_input: list[dict]) -> pd.DataFrame:
    """Extract contents from list of clozemaster collection pages

    Each dict is a page containing a list of collectionClozeSentences.

    Each output row has the contents of one sentence, including "text",
    "translation", "cloze" (detected from {{ }} via regex), and "collection"
    (extracted from "collection.name").
    """
    df = pd.json_normalize(
        raw_input,
        record_path="collectionClozeSentences",
        meta=["collection"],
    )
    df["collection"] = df["collection"].apply(lambda x: x["name"])
    cloze_finder = r"\{\{(?P<cloze>\w*)\}\}"
    df["cloze"] = df["text"].str.extract(cloze_finder)
    return df
