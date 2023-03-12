import argparse

import openai

from clozify_llm.constants import MAX_TOKENS, MODEL, STARTING_MESSAGE, TEMPERATURE


def get_args() -> argparse.Namespace:
    """Get command-line args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input text file")
    parser.add_argument("-o", "--output", required=True, help="output csv file")
    args = parser.parse_args()
    return args


def get_inputs(input_loc: str) -> list[str]:
    """Extract list of input strings from specified text file location"""
    with open(input_loc, "r") as f:
        inputs = f.read().splitlines()
    return inputs


def write_output(responses: list[dict], output_loc: str):
    """Given a list of responses from OpenAI ChatCompletion API, write output to file.

    This is expected to be CSV, but depends on how cooperative ChatGPT is being.
    """
    output_str = ""
    for response in responses:
        response_msg = response["choices"][0]["message"]
        response_content = response_msg["content"]
        output_str += response_content + "\n"

    with open(output_loc, "w") as f:
        f.write(output_str)


def make_chat_params(input_word: str) -> dict:
    """Assemble parameters for openai.ChatCompletion request"""
    prompt_message = {"role": "user", "content": f"Input: {input_word}"}
    messages = STARTING_MESSAGE + [prompt_message]
    return {"model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "messages": messages}


def cli():
    args = get_args()
    inputs = get_inputs(args.input)
    responses = []
    for input_word in inputs:
        chat_params = make_chat_params(input_word)
        response = openai.ChatCompletion.create(**chat_params)
        print(f"response for {input_word} received, usage {response.get('usage')}")
        responses.append(response)
    write_output(responses, args.output)
    print(f"wrote {len(responses)} responses to {args.output}")
