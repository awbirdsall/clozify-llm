"""predict.py Perform model inference
"""
import openai
from openai.openai_object import OpenAIObject
from tenacity import retry, stop_after_attempt, wait_random_exponential

from clozify_llm.constants import END_STR
from clozify_llm.utils import format_prompt


class Completer:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def get_completion_response(self, word: str, defn: str, **kwargs) -> OpenAIObject:
        """Get completion response from word and definition

        Includes formatting prompt assumed to be in same way that completion model was fine-tuned.
        """
        prompt = format_prompt(word, defn)
        completion_kwargs = {
            "max_tokens": 200,
            "temperature": 0.2,
        }
        completion_kwargs.update(**kwargs)
        completion = self._get_completion_with_backoff(
            model=self.model_id,
            prompt=prompt,
            stop=END_STR,
            **completion_kwargs,
        )
        return completion

    def get_cloze_text(self, word: str, defn: str) -> str:
        """Get single cloze completion text from OpenAIObject"""
        completion = self.get_completion_response(word, defn)
        cloze_response = completion["choices"][0]["text"]
        print(f"response for {word} received, total usage {completion.get('usage').get('total_tokens')}")
        return cloze_response

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _get_completion_with_backoff(self, model: str, prompt: str, stop: str, **kwargs):
        """Wrap openai.Completion.create with retry to handle rate limit errors"""
        return openai.Completion.create(model=model, prompt=prompt, stop=stop, **kwargs)
