"""utils.py Utility functions
"""
from time import sleep

import openai
from openai.error import RateLimitError

from clozify_llm.constants import (
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_EMB_ENG,
    STARTING_MESSAGE,
)


def make_chat_params(
    input_word: str,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_CHAT_TEMPERATURE,
    max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
) -> dict:
    """Assemble parameters for openai.ChatCompletion request"""
    prompt_message = {"role": "user", "content": f"Input: {input_word}"}
    messages = STARTING_MESSAGE + [prompt_message]
    return {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}


def get_emb(x, embedding_engine=DEFAULT_EMB_ENG):
    """Get embedding response for input from OpenAI API"""
    resp = openai.Embedding.create(input=x, engine=embedding_engine)
    return resp


def get_embs(xs: list[str]) -> list[list[float]]:
    """Get embedding for list of inputs, with handling of rate limiting."""
    embs = []
    tokens = 0
    for x in xs:
        got_resp = False
        num_attempts = 0
        while not got_resp and num_attempts < 3:
            num_attempts += 1
            try:
                resp = get_emb(x)
                got_resp = True
                print(f"INFO - got response for {x}")
            except RateLimitError:
                print("INFO - hit rate limit, waiting and retrying")
                sleep(5)
        if got_resp:
            emb = resp["data"][0]["embedding"]
            embs.append(emb)
            tokens += resp["usage"]["total_tokens"]
            sleep(1)  # rate limited to 60 / min
        else:
            print(f"ERROR - could not retrieve embedding for {x}")
    print(f"INFO - Got {len(embs)} embeddings, total token usage {tokens}")
    return embs


def format_prompt(word: str, definition: str) -> str:
    return f"{word.strip()}\n{definition.strip()}\n\n###\n\n"


def format_completion(text: str, translation: str, cloze: str) -> str:
    return f" {text},{translation},{cloze} END"
