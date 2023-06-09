"""constants.py Constants for use in project"""

STARTING_MESSAGE = [
    {
        "role": "user",
        "content": "Create a training sentence for a German vocabulary word. Also include the English translation. The response should be a CSV formatted string.",
    },
    {
        "role": "user",
        "content": "Input:einheimisch",
    },
    {
        "role": "assistant",
        "content": '"Die einheimische Jägerschaft hält sich zurück.","The local hunters hold back.","einheimische"',
    },
]
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
DEFAULT_CHAT_TEMPERATURE = 0.2
DEFAULT_CHAT_MAX_TOKENS = 256
DEFAULT_EMB_ENG = "text-embedding-ada-002"
CLOZE_COL = "cloze"
WORD_COL = "word"
DEFN_COL = "defn"
DEFAULT_COMPLETION_MODEL = "curie"
END_STR = " END"
PROMPT_SEPARATOR = "\n\n###\n\n"
QUOTECHAR = '"'
