import json
import os
from getpass import getpass
from pathlib import Path

import click
import openai
import pandas as pd

from clozify_llm.embed import add_emb
from clozify_llm.extract.extract_cloze import extract_cloze
from clozify_llm.extract.extract_wortschatz import get_all_vocab_from_course_request

# from clozify_llm.utils import make_chat_params
# from clozify_llm.join import join_emb_sim, clean_join

# import openai


@click.group()
def cli():
    pass


# @click.command()
# def make_cloze():
#     args = get_args()
#     inputs = get_inputs(args.input)
#     responses = []
#     for input_word in inputs:
#         chat_params = make_chat_params(input_word)
#         response = openai.ChatCompletion.create(**chat_params)
#         click.echo(f"response for {input_word} received, usage {response.get('usage')}")
#         responses.append(response)
#     write_output(responses, args.output)
#     click.echo(f"wrote {len(responses)} responses to {args.output}")


# @click.command()
# def prep_join():
#     joiner = Joiner()
#     join_cloze_vocab()


@cli.command()
@click.argument("url")
@click.option("--output", default="wortschatz.csv", help="Output location.")
@click.option("--staging", default="tmp", help="Directory to save intermediate html.")
def fetch(url, output, staging):
    words = get_all_vocab_from_course_request(url, staging)
    words.to_csv(output)
    print(f"wrote {len(words)} to {output}")


@cli.command()
@click.argument("json_file")
@click.option("--output", default="output.csv", help="Output CSV file.")
def parse(json_file, output):
    with open(json_file, "r") as f:
        data = json.load(f)
    clozes = extract_cloze(data)
    clozes.to_csv(output)
    print(f"wrote {len(clozes)} to {output}")


@cli.command()
@click.argument("csv_files", nargs=-1, type=click.Path(exists=True))
@click.option("--output", default="output", help="Output dir.")
def embed(csv_files, output):
    if os.getenv("OPENAI_API_KEY") is None:
        openai.api_key = getpass()
    Path(output).mkdir(exist_ok=True, parents=True)
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        df = pd.read_csv(csv_path)
        df_emb = add_emb(df)
        output_csv = Path(output) / f"{csv_path.stem}-embeds.csv"
        df_emb.to_csv(output_csv)
        print(f"wrote {len(df_emb)} to {output_csv}")
        # Add your logic to embed the data and write to CSV
        pass


@cli.command()
@click.argument("csv_file1", type=click.Path(exists=True))
@click.argument("csv_file2", type=click.Path(exists=True))
@click.option("--output", default="output.csv", help="Output CSV file.")
def match(csv_file1, csv_file2, output):
    # Add your logic to match the input CSV files and write to CSV
    pass


@cli.command()
@click.argument("csv_file1", type=click.Path(exists=True))
@click.argument("csv_file2", type=click.Path(exists=True))
@click.option("--output", default="output.csv", help="Output CSV file.")
def fix(csv_file1, csv_file2, output):
    # Add your logic to fix the input CSV files and write to CSV
    pass


@cli.command()
@click.argument("csv_file", type=click.Path(exists=True))
def finetune(csv_file):
    # Add your logic to finetune the model with the input CSV file
    # and display the identifier of the fine-tuned model
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", default="output.csv", help="Output CSV file.")
def complete(input, output):
    # Add your logic to complete the input single word or text file
    # and return the result as a CSV-formatted string
    pass


# cli.add_command(make_cloze)
# cli.add_command(prep_join)


# def get_args() -> argparse.Namespace:
#     """Get command-line args"""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", required=True, help="input text file")
#     parser.add_argument("-o", "--output", required=True, help="output csv file")
#     args = parser.parse_args()
#     return args


# def get_inputs(input_loc: str) -> list[str]:
#     """Extract list of input strings from specified text file location"""
#     with open(input_loc, "r") as f:
#         inputs = f.read().splitlines()
#     return inputs


# def write_output(responses: list[dict], output_loc: str):
#     """Given a list of responses from OpenAI ChatCompletion API, write output to file.

#     This is expected to be CSV, but depends on how cooperative ChatGPT is being.
#     """
#     output_str = ""
#     for response in responses:
#         response_msg = response["choices"][0]["message"]
#         response_content = response_msg["content"]
#         output_str += response_content + "\n"

#     with open(output_loc, "w") as f:
#         f.write(output_str)


if __name__ == "__main__":
    cli()
