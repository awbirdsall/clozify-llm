"""predict.py Perform model inference
"""
import openai
from openai.openai_object import OpenAIObject

from clozify_llm.constants import END_STR
from clozify_llm.utils import format_prompt


class Completer:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def get_completion_response(self, word: str, defn: str, **kwargs) -> OpenAIObject:
        """Get completion response from word and definition

        Must format prompt in same way that completion model was fine-tuned.
        """
        prompt = format_prompt(word, defn)
        completion_kwargs = {
            "max_tokens": 200,
            "temperature": 0.2,
        }
        completion_kwargs.update(**kwargs)
        completion = openai.Completion.create(
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
        return cloze_response
