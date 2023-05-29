"""utils.py Utility functions
"""
import csv
from io import StringIO

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from clozify_llm.constants import DEFAULT_EMB_ENG, END_STR, PROMPT_SEPARATOR, QUOTECHAR


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_emb(x, embedding_engine=DEFAULT_EMB_ENG):
    """Get embedding response for input from OpenAI API

    Wrap with retry to handle rate limit errors"""
    resp = openai.Embedding.create(input=x, engine=embedding_engine)
    return resp


def get_embs(xs: list[str]) -> list[list[float]]:
    """Get embedding for list of inputs, with handling of rate limiting."""
    embs = []
    tokens = 0
    for x in xs:
        resp = get_emb(x)
        emb = resp["data"][0]["embedding"]
        embs.append(emb)
        tokens += resp["usage"]["total_tokens"]
    print(f"INFO - Got {len(embs)} embeddings, total token usage {tokens}")
    return embs


def format_prompt(word: str, definition: str) -> str:
    """Format a prompt for fine-tuning following recommended practices

    - End with fixed separator to inform model where prompt end and completion begins

    References
    ----------
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """
    return f"{word.strip()}\n{definition.strip()}{PROMPT_SEPARATOR}"


def format_completion(text: str, translation: str, cloze: str) -> str:
    """Format a completion for fine-tuning following recommended practices

    - Start with whitespace
    - End with fixed stop sequence

    To help make the output CSV more easily parseable, also set QUOTE_ALL and use escapechar for quotation marks.

    References
    ----------
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """
    with StringIO() as buf:
        writer = csv.writer(buf, quoting=csv.QUOTE_ALL, quotechar=QUOTECHAR, doublequote=False, escapechar="\\")
        writer.writerow([text, translation, cloze])
        csv_str = buf.getvalue().strip()
    return " " + csv_str + END_STR
