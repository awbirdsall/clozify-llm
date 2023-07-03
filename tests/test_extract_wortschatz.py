from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from clozify_llm.constants import DEFN_COL, WORD_COL
from clozify_llm.extract.extract_wortschatz import (
    extract_script,
    get_all_vocab_from_course_request,
    lessons_from_course,
    load_lesson,
    words_from_wortschatz_html,
    wortschatz_html_from_id,
)


@pytest.fixture
def course_html():
    return """
        <html>
            <body>
                <ul>
                    <li class="lesson-item"><a href="/de/lesson1/l-123">Lesson 1</a></li>
                    <li class="lesson-item"><a href="/de/lesson2/l-456">Lesson 2</a></li>
                </ul>
            </body>
        </html>
    """


@pytest.fixture
def ws_html():
    return """
        <html>
            <body>
                <div class="knowledge-wrapper">
                    <div>
                        <strong>Word 1</strong>
                        <p>Definition 1</p>
                    </div>
                    <div>
                        <strong>Word 2</strong>
                        <p>Definition 2</p>
                    </div>
                </div>
            </body>
        </html>
    """


@pytest.fixture
def mock_requests_get():
    response_mock = mock.Mock(content=b"dummy content")
    get_mock = mock.Mock(return_value=response_mock)
    with mock.patch("requests.get", get_mock):
        yield get_mock


def test_get_all_vocab_from_course_request(course_html, ws_html, mock_requests_get, tmp_path):
    """Test complete get_all_vocab_from_course_request behavior with mocked get requests"""
    course_response_mock = mock.Mock(content=course_html.encode("utf-8"))
    lesson_response_mock = mock.Mock(content=ws_html.encode("utf-8"))

    # Return the lesson html unless the course html is requested
    dummy_course_url = "https://example.com/course"

    def mock_get(url):
        if url == dummy_course_url:
            return course_response_mock
        else:
            return lesson_response_mock

    mock_requests_get.side_effect = mock_get

    expected_df = pd.DataFrame({WORD_COL: ["Word 1", "Word 2"], DEFN_COL: ["Definition 1", "Definition 2"]})

    result_df = get_all_vocab_from_course_request(dummy_course_url, str(tmp_path))

    assert_frame_equal(result_df, expected_df)
    mock_requests_get.assert_any_call(dummy_course_url)

    # Verify that lesson URLs were requested
    expected_lesson_urls = [
        "https://learngerman.dw.com/de/lesson1/l-123/lv",
        "https://learngerman.dw.com/de/lesson2/l-456/lv",
    ]
    for lesson_url in expected_lesson_urls:
        mock_requests_get.assert_any_call(lesson_url)


def test_extract_script(course_html, ws_html, mock_requests_get):
    """Test behavior of extract_script() with patched value for load_lesson() response

    Note this reuses the same return value for load_lesson regardless of lesson_id
    """
    expected_df = pd.DataFrame({WORD_COL: ["Word 1", "Word 2"], DEFN_COL: ["Definition 1", "Definition 2"]})

    with mock.patch("clozify_llm.extract.extract_wortschatz.load_lesson", return_value=ws_html):
        result_df = extract_script(course_html, Path("/tmp"))

    assert_frame_equal(result_df, expected_df)


def test_lessons_from_course(course_html):
    expected_lessons = ["lesson1/l-123", "lesson2/l-456"]

    result_lessons = lessons_from_course(course_html)

    assert result_lessons == expected_lessons


def test_load_lesson(mock_requests_get, ws_html, tmp_path):
    """Test load_lesson behavior when local file does not initially exist

    Mock request response with expected lesson html"""
    response_mock = mock.Mock(content=ws_html.encode("utf-8"))
    mock_requests_get.return_value = response_mock
    local_dir = tmp_path / "lessons"
    local_dir.mkdir()
    lesson_id = "lesson1/l-123"
    expected_file_path = local_dir / "lesson1_l-123.html"

    with mock.patch("builtins.open", mock.mock_open()) as mock_open:
        result_html = load_lesson(lesson_id, local_dir)

    assert result_html == ws_html
    mock_requests_get.assert_called_once_with("https://learngerman.dw.com/de/lesson1/l-123/lv")
    mock_open.assert_called_once_with(expected_file_path, "w")
    handle = mock_open()
    handle.write.assert_called_once_with(ws_html)


def test_wortschatz_html_from_id(mock_requests_get):
    expected_html = "<html>dummy content</html>"
    response_mock = mock.Mock(content=expected_html.encode("utf-8"))
    mock_requests_get.return_value = response_mock

    result_html = wortschatz_html_from_id("lesson1/l-123")

    assert result_html == expected_html
    mock_requests_get.assert_called_once_with("https://learngerman.dw.com/de/lesson1/l-123/lv")


def test_words_from_wortschatz_html(ws_html):
    """Test behavior of words_from_wortschatz_html on example html str"""
    expected_words = [
        {WORD_COL: "Word 1", DEFN_COL: "Definition 1"},
        {WORD_COL: "Word 2", DEFN_COL: "Definition 2"},
    ]

    result_words = words_from_wortschatz_html(ws_html)

    assert result_words == expected_words
