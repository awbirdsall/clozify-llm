"""finetune.py Functionality to finetune completion LLM with training data
"""
import json

import openai
import pandas as pd

from clozify_llm.constants import (
    CLOZE_COL,
    DEFAULT_COMPLETION_MODEL,
    DEFN_COL,
    WORD_COL,
)


class FineTuner:
    def __init__(self, df: pd.DataFrame, training_data_path: str, model: str = DEFAULT_COMPLETION_MODEL):
        self.df = df
        self.training_data_path = training_data_path
        self.model = model

    @staticmethod
    def format_prompt(word: str, definition: str) -> str:
        return f"{word.strip()}\n{definition.strip()}\n\n###\n\n"

    @staticmethod
    def format_completion(text: str, translation: str, cloze: str) -> str:
        return f" {text},{translation},{cloze} END"

    def create_dataset(self) -> list[dict]:
        train_rows = []
        print(f"columns {self.df.columns}")
        to_iter = self.df[[WORD_COL, DEFN_COL, "text", "translation", CLOZE_COL]]
        for row in to_iter.itertuples():
            print(f"formatting row {row}")
            prompt = self.format_prompt(getattr(row, WORD_COL), getattr(row, DEFN_COL))
            completion = self.format_completion(row.text, row.translation, getattr(row, CLOZE_COL))
            train_rows.append({"prompt": prompt, "completion": completion})
        return train_rows

    def start_finetuning(self):
        dataset = self.create_dataset()
        with open(self.training_data_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

        with open(self.training_data_path, "r") as f:
            file_response = openai.File.create(file=f, purpose="fine-tune")
        print(f"File.create with id {file_response.id}")

        ft_response = openai.FineTune.create(training_file=file_response.id, model=self.model)
        return ft_response
