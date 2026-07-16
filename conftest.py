"""Pytest support for the evals test suite.

``evals.utils`` constructs an ``OpenAI`` client at import time from
``OPENAI_API_KEY`` / ``OPENAI_ORGANIZATION_ID``. The tests in this repo exercise
pure logic (no network), so we seed placeholder credentials before any test
module imports ``evals``. We also make sure the repository root (which holds the
importable ``evals`` namespace package) is on ``sys.path`` regardless of how
pytest is invoked.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("OPENAI_API_KEY", "test-placeholder-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-placeholder-org")
