"""Pytest path + environment shim for the deep-research evals repo.

Two concerns, both about *import* time rather than test execution:

* ``evals/utils.py`` constructs an ``OpenAI`` client at module import time and
  reads ``OPENAI_API_KEY`` / ``OPENAI_ORGANIZATION_ID`` straight from the
  environment. The constructor is lazy -- it never makes a network call -- so
  placeholder values are enough to let the ``evals`` package import under a
  test runner that has no credentials. The tests in this repo exercise pure
  analysis helpers offline and never call the API.
* The repo has no installed package metadata (no ``pyproject.toml`` /
  ``setup.py``), so make the repository root importable for ``import evals``
  regardless of how pytest is invoked.
"""

import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org-not-used")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
