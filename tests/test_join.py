import pandas as pd

from clozify_llm.join import Joiner


def test_joiner():
    joiner = Joiner(pd.DataFrame(), pd.DataFrame(), "", "")
