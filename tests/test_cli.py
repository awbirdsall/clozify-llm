"""Test cli.py"""

from ast import literal_eval
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner
from pandas.testing import assert_frame_equal

from clozify_llm.cli import (
    chat,
    cli,
    complete,
    embed,
    fetch,
    finetune,
    fix,
    get_help_recursive,
    match,
    parse,
)


@pytest.fixture
def runner():
    return CliRunner()


@patch("clozify_llm.cli.ChatCompleter")
@patch("clozify_llm.cli.getpass")
def test_chat_from_word(mock_getpass, mock_completer, runner, tmp_path):
    """Test cli.chat with word input and output written to tmp file location

    Mock return of completer call.
    """
    mock_completer_instance = mock_completer.return_value
    mock_completer_instance.get_cloze_text.return_value = "my completion"

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        output_loc = f"{td}/output.csv"
        result = runner.invoke(chat, ["Wort", "--output", output_loc])
        result_contents = Path(output_loc).read_text()

    assert result.exit_code == 0
    assert result.output == f"wrote 1 responses to {output_loc}\n"
    assert result_contents == "my completion\n"

    mock_completer_instance.get_cloze_text.assert_called_once()


@patch("clozify_llm.cli.get_all_vocab_from_course_request")
def test_fetch(mock_get_all_vocab, runner, tmp_path):
    """Test cli.fetch with mocked get_all_vocab call and output written to tmp file

    Mocked call only returns result dataframe, does not include side effect of saving intermediate html
    """
    df = pd.DataFrame({"word": ["apple", "banana", "pear"]})
    mock_get_all_vocab.return_value = df

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        output_loc = f"{td}/wortschatz.csv"
        result = runner.invoke(fetch, ["courseurl.com", "--output", output_loc])
        result_contents = pd.read_csv(output_loc)

    assert result.exit_code == 0
    assert result.output == f"wrote {len(df)} to {output_loc}\n"
    assert_frame_equal(result_contents, df)

    mock_get_all_vocab.assert_called_once()


@patch("clozify_llm.cli.extract_cloze")
def test_parse(mock_extract_cloze, runner, tmp_path):
    """Test cli.parse with mocked extract_cloze call and output written to tmp file

    Input file location is created with dummy data but does not correspond to output
    """
    dummy_data = "[{}]"
    df = pd.DataFrame({"cloze": ["Wort"], "collection": ["my-collection"]})
    mock_extract_cloze.return_value = df

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        input_path = Path(td) / "input.json"
        input_path.write_text(dummy_data)
        output_loc = f"{td}/output.csv"
        result = runner.invoke(parse, [str(input_path), "--output", output_loc])
        result_contents = pd.read_csv(output_loc)

    assert result.exit_code == 0
    assert result.output == f"wrote {len(df)} to {output_loc}\n"
    assert_frame_equal(result_contents, df)

    mock_extract_cloze.assert_called_once()


@patch("clozify_llm.cli.add_emb")
@patch("clozify_llm.cli.getpass")
def test_embed(mock_getpass, mock_add_emb, runner, tmp_path):
    """Test cli.embed with mocked add_emb call and output written to tmp file

    Input file location is created with dummy data but does not correspond to output

    Test only with a single csv input/output
    """
    df = pd.DataFrame({"cloze": ["Wort"]})
    df_emb = df.copy()
    embedding_col = pd.Series(data=[[0, 1, 0.5]])
    df_emb["cloze_embedding"] = embedding_col
    mock_add_emb.return_value = df_emb

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        input_path = Path(td) / "input.csv"
        df.to_csv(input_path, index=False)
        output_dir = f"{td}/output"
        result = runner.invoke(embed, [str(input_path), "--output", output_dir])
        expected_output_csv = f"{output_dir}/input-embeds.csv"
        result_contents = pd.read_csv(expected_output_csv)

    assert result.exit_code == 0
    assert result.output == f"wrote {len(df_emb)} to {expected_output_csv}\n"
    # Check output contains str that can be eval back to list
    result_df = result_contents.copy()
    result_df["cloze_embedding"] = result_df["cloze_embedding"].apply(lambda x: literal_eval(x))
    assert_frame_equal(result_df, df_emb)

    mock_add_emb.assert_called_once()


@patch("clozify_llm.cli.Joiner")
def test_match(mock_joiner, runner, tmp_path):
    """Test cli.match with mocked Joiner and output written to tmp file"""
    df_cloze = pd.DataFrame({"cloze": ["Wort"]})
    df_vocab = pd.DataFrame({"word": ["word"]})
    df_join = pd.concat([df_cloze, df_vocab], axis=1)
    mock_joiner_instance = mock_joiner.return_value
    mock_joiner_instance.join_emb_sim.return_value = df_join

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        input_cloze = f"{td}/cloze.csv"
        input_vocab = f"{td}/vocab.csv"
        df_cloze.to_csv(input_cloze, index=False)
        df_vocab.to_csv(input_vocab, index=False)
        output_loc = f"{td}/output.csv"
        result = runner.invoke(match, [input_cloze, input_vocab, "--output", output_loc])
        result_contents = pd.read_csv(output_loc)

    assert result.exit_code == 0
    assert result.output == f"wrote candidate join len {len(df_join)} to {output_loc}\n"
    assert_frame_equal(result_contents, df_join)

    mock_joiner_instance.join_emb_sim.assert_called_once()


@patch("clozify_llm.cli.Joiner")
def test_fix(mock_joiner, runner, tmp_path):
    """Test cli.fix with mocked Joiner and output written to tmp file"""
    df_join = pd.DataFrame({"word": ["Apfel"]})
    mock_joiner_instance = mock_joiner.return_value
    mock_joiner_instance.clean_join_from_review.return_value = df_join

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        input_candidate = f"{td}/candidate.csv"
        input_review = f"{td}/review.csv"
        input_vocab = f"{td}/vocab.csv"
        for input_path in [input_candidate, input_review, input_vocab]:
            pd.DataFrame().to_csv(input_path)
        output_loc = f"{td}/output.csv"
        result = runner.invoke(fix, [input_candidate, input_review, input_vocab, "--output", output_loc])
        result_contents = pd.read_csv(output_loc)

    assert result.exit_code == 0
    assert result.output == f"wrote corrected join len {len(df_join)} to {output_loc}\n"
    assert_frame_equal(result_contents, df_join)

    mock_joiner_instance.clean_join_from_review.assert_called_once()


@patch("clozify_llm.cli.FineTuner")
@patch("clozify_llm.cli.getpass")
def test_finetune(mock_getpass, mock_finetuner, runner, finetune_create_response, tmp_path):
    """Test cli.finetune with mocked FineTuner and output written to tmp file location"""
    mock_finetuner_instance = mock_finetuner.return_value
    mock_finetuner_instance.start_finetuning.return_value = finetune_create_response

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        input_csv = f"{td}/input.csv"
        pd.DataFrame().to_csv(input_csv)
        output_loc = f"{td}/output.csv"
        result = runner.invoke(finetune, [input_csv, output_loc])

    assert result.exit_code == 0
    assert result.output == f"FineTune job created\n{finetune_create_response}\n"

    mock_finetuner_instance.start_finetuning.assert_called_once()


@patch("clozify_llm.cli.Completer")
@patch("clozify_llm.cli.getpass")
def test_complete_from_word_defn(mock_getpass, mock_completer, runner, tmp_path):
    """Test cli.complete with word and defn input, mocked Completer and output written to tmp file location"""
    mock_completer_instance = mock_completer.return_value
    dummy_completion = "my completion,has,fields"
    mock_completer_instance.get_cloze_text.return_value = dummy_completion

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        input_csv = f"{td}/input.csv"
        pd.DataFrame().to_csv(input_csv)
        output_loc = f"{td}/output.csv"
        result = runner.invoke(complete, ["myword", "my defn", "--model_id", "my-model", "--output", output_loc])
        result_contents = Path(output_loc).read_text()

    assert result.exit_code == 0
    assert result.output == f"wrote 1 to {output_loc}\n"
    assert result_contents == f"{dummy_completion}\n"

    mock_completer_instance.get_cloze_text.assert_called_once()


def test_get_help_recursive_runs(runner):
    """Test ability to call get_help_recursive on entire cli"""
    result = get_help_recursive(cli)
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result.splitlines()) > 20
