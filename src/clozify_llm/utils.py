"""utils.py Utility functions
"""
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from clozify_llm.constants import DEFAULT_EMB_ENG


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
    return f"{word.strip()}\n{definition.strip()}\n\n###\n\n"


def format_completion(text: str, translation: str, cloze: str) -> str:
    return f" {text},{translation},{cloze} END"
