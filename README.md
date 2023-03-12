# clozify-llm

A tiny tool to help generate sentences for Clozemaster language learning using large language models (LLM).

Wraps around the ChatGPT API provided by OpenAI.

## Why?

I like using Clozemaster. I like the language resources on DW's Learn German site. I like adding words from the word lists in the DW Learn German lessons but the words are often too specialized to have existing Clozes.

I can trawl the web to find examples. (Thank you, search bars on German public radio websites!) And sometimes the searching for a good example is a helpful part of the learning itself. But sometimes I also just want to add sentences to Clozemaster.

The idea here is prompting ChatGPT for each word in a list and requesting a response that's already in the CSV format Clozemaster accepts for batch uploads.

## How?

You're gonna need an OpenAI account and API key (assumed to be available at standard `OPENAI_API_KEY` env var). It's gonna cost ya once you run through the free credits.

### Install

```bash
$ poetry install
```

This installs the dependencies in a `poetry` environment as well as a simple CLI named `clozify`.

Verify installation:

```bash
$ poetry run clozify --help
```

### Use

```bash
$ echo "regional" >> vocab.txt
$ poetry run clozify -i vocab.txt -o clozes.csv
response for regional received, usage {
  "completion_tokens": 30,
  "prompt_tokens": 99,
  "total_tokens": 129
}
wrote 1 responses to tmp/clozes.csv
$ cat clozes.csv
Die Firma hat sich auf die Herstellung regionaler Produkte spezialisiert.,The company specializes in the production of regional products.,regional
```

## Limitations

This is relying on machine translation so all limitations there apply. The output might have subtle issues with grammar, idiomatic usage, etc. The assumption is the output will receive manual human review for these issues before being added to a flashcard set.

The output appears to struggle with identifying when the input word was inflected in the output. This has an impact on how Clozemaster ingests the list of sentences and identifies the cloze. This could probably be cleaned up with some postprocessing rules, but again, manual human review can also catch.

Cleaning of the output might be necessary if the CSV format gets mangled in some responses.

There is no content filtering applied for potentially objectionable material.

Certain parameters are hard-coded (model, temperature, etc.).

The input does also not provide a contextual definition to use, though this is something I've tried out and would like to implement.

## Development

Environment managed with `poetry` v1.4.

Autoformatting checks managed with `pre-commit`.

## References

This was inspired by Matt Webb's description of his Braggoscope project ([about page](https://genmon.github.io/braggoscope/about), [blog writeup](https://interconnected.org/home/2023/02/07/braggoscope)), which uses ChatGPT to extract data from *In Our Time* shownotes and perform some other tasks (embeddings).

Robin Sloan's [gloss](https://www.robinsloan.com/lab/phase-change/) also provided a nudge:

> Where the GPT-alikes are concerned, a question that’s emerging for me is:
>
> *What could I do with a universal function — a tool for turning just about any X into just about any Y with plain language instructions?*
