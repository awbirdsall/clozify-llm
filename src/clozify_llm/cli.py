import json
import os
from getpass import getpass
from pathlib import Path

import click
import openai
import pandas as pd

from clozify_llm.constants import DEFN_COL, WORD_COL
from clozify_llm.embed import add_emb
from clozify_llm.extract.extract_cloze import extract_cloze
from clozify_llm.extract.extract_wortschatz import get_all_vocab_from_course_request
from clozify_llm.finetune import FineTuner
from clozify_llm.join import Joiner
from clozify_llm.predict import ChatCompleter, Completer


@click.group()
def cli():
    """Use LLMs to generate cloze sentences."""
    pass


@cli.command()
@click.argument("word", required=False)
@click.option("-f", "--file", type=click.Path(exists=True), required=False, help="Read input from file")
@click.option("-o", "--output", type=click.Path(allow_dash=True), default="-", help="Output location")
def chat(word, file, output):
    """Generate clozes using a chat model

    Read WORD or each line in FILE, generate a cloze, and write to OUTPUT.
    """
    if os.getenv("OPENAI_API_KEY") is None:
        openai.api_key = getpass()
    completer = ChatCompleter()
    if word:
        inputs = [word]
    elif file:
        with click.open_file(file, "r") as f:
            inputs = f.read().splitlines()
    else:
        click.echo("No input provided. Please provide either an input string or a file path")
    responses = []
    for input_word in inputs:
        completion = completer.get_cloze_text(input_word, defn="")
        responses.append(completion)

    write_output(responses, output)
    click.echo(f"wrote {len(responses)} responses to {output}")


@cli.group()
def prep():
    """Prepare training data for model fine-tuning."""
    pass


@prep.command()
@click.argument("url")
@click.option("--output", default="wortschatz.csv", help="Output location.")
@click.option("--staging", default="tmp", help="Directory to save intermediate html.")
def fetch(url, output, staging):
    """Get vocabulary from a course"""
    words = get_all_vocab_from_course_request(url, staging)
    words.to_csv(output)
    print(f"wrote {len(words)} to {output}")


@prep.command()
@click.argument("json_file")
@click.option("--output", default="output.csv", help="Output CSV file.")
def parse(json_file, output):
    """Extract clozes from scraped json data

    Used as part of the training data generation process
    """
    with click.open_file(json_file, "r") as f:
        data = json.load(f)
    clozes = extract_cloze(data)
    clozes.to_csv(output)
    print(f"wrote {len(clozes)} to {output}")


@prep.command()
@click.argument("csv_files", nargs=-1, type=click.Path(exists=True))
@click.option("--output", default="output", help="Output dir.")
def embed(csv_files, output):
    """Get embeddings for the word or cloze in the input

    Used as part of the training data generation process
    """
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


@prep.command()
@click.argument("cloze_csv", type=click.Path(exists=True))
@click.argument("vocab_csv", type=click.Path(exists=True))
@click.option("--output", default="output.csv", help="Output CSV file.")
def match(cloze_csv, vocab_csv, output):
    """Join cloze and vocab data based on embedding similarities

    Used as part of the training data generation process
    """
    df_cloze = pd.read_csv(cloze_csv)
    df_vocab = pd.read_csv(vocab_csv)
    joiner = Joiner(df_cloze, df_vocab)
    joined = joiner.join_emb_sim()
    joined.to_csv(output)
    print(f"wrote candidate join len {len(joined)} to {output}")


@prep.command()
@click.argument("candidate_join", type=click.Path(exists=True))
@click.argument("manual_review", type=click.Path(exists=True))
@click.argument("vocab_csv", type=click.Path(exists=True))
@click.option("--output", default="output.csv", help="Output CSV file.")
def fix(candidate_join, manual_review, vocab_csv, output):
    """Update candidate training data based on manual review

    Used as part of the training data generation process
    """
    df_candidate = pd.read_csv(candidate_join)
    df_manual = pd.read_csv(manual_review)
    df_vocab = pd.read_csv(vocab_csv)
    joiner = Joiner(df_candidate, df_vocab)
    fixed = joiner.clean_join_from_review(df_candidate, df_manual)
    fixed.to_csv(output)
    print(f"wrote corrected join len {len(fixed)} to {output}.")


@cli.command()
@click.argument("csv_file", type=click.Path(exists=True))
@click.argument("training_data_output")
def finetune(csv_file, training_data_output):
    """
    Start completion model fine-tuning from training data

    Start model fine-tuning using data in CSV_FILE written to TRAINING_DATA_OUTPUT in the format
    that is uploaded for fine-tuning. Details of the FineTune job is printed.
    """
    if os.getenv("OPENAI_API_KEY") is None:
        openai.api_key = getpass()
    df = pd.read_csv(csv_file)
    fine_tuner = FineTuner(df, training_data_output)
    ft_response = fine_tuner.start_finetuning()
    click.echo("FineTune job created")
    click.echo(ft_response)


@cli.command()
@click.argument("word", required=False)
@click.argument("defn", required=False)
@click.option("-f", "--file", type=click.Path(exists=True), required=False, help="Input CSV file")
@click.option("-m", "--model_id", required=True, help="Fine tuned completion model")
@click.option("-o", "--output", type=click.Path(allow_dash=True), default="-", help="Output CSV file.")
def complete(word, defn, file, model_id, output):
    """Generate clozes using a completion model

    Read WORD and DEFN or each line in FILE, provide to fine-tuned MODEL_ID, and write to OUTPUT.
    """
    if os.getenv("OPENAI_API_KEY") is None:
        openai.api_key = getpass()
    completer = Completer(model_id)
    if word and defn:
        df_inputs = pd.DataFrame({WORD_COL: [word], DEFN_COL: [defn]})
    elif file:
        df_inputs = pd.read_csv(file)
    else:
        click.echo("No input provided. Please provide either an input word and defn or a file path")
    cloze_texts = []
    for row in df_inputs.itertuples():
        word = getattr(row, WORD_COL)
        defn = getattr(row, DEFN_COL)
        completion = completer.get_cloze_text(word, defn)
        cloze_texts.append(completion)
    write_output(cloze_texts, output)
    click.echo(f"wrote {len(cloze_texts)} to {output}")


def write_output(cloze_texts: list[str], output_loc: str):
    """Given a list of texts, write to file at specified location.

    This is expected to be CSV, but depends on how cooperative ChatGPT is being.
    """
    with click.open_file(output_loc, "w") as f:
        f.writelines(cloze_line + "\n" for cloze_line in cloze_texts)


def get_help_recursive(command, parent_name=""):
    """Helper function to generate help strings for all commands"""
    ctx = click.Context(command)
    if command.name == "cli":
        cmd_name = "clozify"
    else:
        cmd_name = f"{parent_name} {command.name}"
    help_string = f"$ {cmd_name} --help\n" + command.get_help(ctx) + "\n"
    if hasattr(command, "commands"):
        for subcommand in command.commands.values():
            help_string += "\n" + get_help_recursive(subcommand, cmd_name)
    return help_string


if __name__ == "__main__":
    cli()
