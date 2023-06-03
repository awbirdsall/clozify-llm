"""test_utils.py Unit testing of utils
"""
from clozify_llm.constants import END_STR
from clozify_llm.utils import format_completion


def test_format_completion():
    csv_str = format_completion("text", 'translation,"quotation",commas', "cloze")
    expected = ' "text","translation,\\"quotation\\",commas","cloze"' + END_STR
    assert csv_str == expected
