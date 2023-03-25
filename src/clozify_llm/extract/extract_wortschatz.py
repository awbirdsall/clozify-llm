"""extract-wortschatz.py helper function to extract word list

Assumes DW Learngerman format"""
from pathlib import Path
from unicodedata import normalize

import pandas as pd
import requests
from bs4 import BeautifulSoup


def extract_script(course_html: str, local_dir: Path) -> pd.DataFrame:
    """Given course page html, collect vocab list from all linked lessons

    Drop fully duplicate entries (want to preserve duplicate words with
    different definitions)

    Returns
    -------
    DataFrame with columns "word" and "def"
    """
    lessons = lessons_from_course(course_html)
    words = []
    for lesson_id in lessons:
        ws_html = load_lesson(lesson_id, local_dir)
        lesson_words = words_from_wortschatz_html(ws_html)
        print(f"{lesson_id} - extracted word count {len(lesson_words)}")
        words.extend(lesson_words)
    df = pd.DataFrame(words).drop_duplicates()
    return df


def lessons_from_course(html_str: str) -> list[str]:
    """Given course page html contents, extract list of lesson ids"""
    soup = BeautifulSoup(html_str, features="html.parser")
    lesson_items = soup.find_all("li", "lesson-item")
    lesson_ids = []
    for li in lesson_items:
        lesson_href = li.a["href"]
        # only need last two parts of href to identify
        lesson_id = "/".join(lesson_href.split("/")[-2:])
        lesson_ids.append(lesson_id)
    return lesson_ids


def load_lesson(lesson_id: str, local_dir: Path) -> str:
    """Load lesson from local dir if present, otherwise request and write"""
    local_path = local_dir / f"{lesson_id.replace('/', '_')}.html"
    if local_path.exists():
        print(f"{lesson_id} -- use {local_path}")
        with open(local_path, "r") as f:
            ws_html = f.read()
    else:
        print(f"{lesson_id} -- request")
        ws_html = wortschatz_html_from_id(lesson_id)
        with open(local_path, "w") as f:
            f.write(ws_html)
        print(f"{lesson_id} -- written to {local_path}")
    return ws_html


def wortschatz_html_from_id(lesson_id: str) -> str:
    """Issue request for html contents of wortschatz page for lesson"""
    response = requests.get(f"https://learngerman.dw.com/de/{lesson_id}/lv")
    html_str = response.content.decode("utf-8")
    return html_str


def words_from_wortschatz_html(html_str: str) -> list[dict[str, str]]:
    """Given Wortschatz page html contents, extract word list

    Normalize string contents to avoid stray \xa0 from non-breaking spaces
    """
    soup = BeautifulSoup(html_str, features="html.parser")
    knowledge = soup.find(class_="knowledge-wrapper")
    kdivs = knowledge.find_all("div")
    words = []
    for kdiv in kdivs:
        word = kdiv.find("strong")
        def_ = kdiv.find("p")
        if word and def_:
            word_text = normalize("NFKD", word.text)
            def_text = normalize("NFKD", def_.text)
            words.append({"word": word_text, "def": def_text})
    return words
