"""Headless smoke tests: actually execute each Streamlit page and assert no
uncaught exception. Pages must degrade gracefully when Layer A artifacts are
absent (they show an info message / st.stop()), so these pass before training too.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _run(path):
    at = AppTest.from_file(path, default_timeout=60).run()
    assert not at.exception, f"{path} raised: {list(at.exception)}"


def test_home_runs():
    _run("app/Home.py")


def test_real_case_runs():
    _run("app/pages/1_Real_Case.py")


def test_illustrative_case_runs():
    _run("app/pages/2_Illustrative_Case.py")


def test_model_card_runs():
    _run("app/pages/3_Model_Card.py")
