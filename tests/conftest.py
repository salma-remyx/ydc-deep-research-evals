"""Pytest configuration shared by the evals test-suite.

``evals.utils`` builds the OpenAI client at import time from
``OPENAI_API_KEY`` / ``OPENAI_ORGANIZATION_ID``, so those must exist before any
evals module is imported. Tests here never hit the network -- they monkeypatch the
structured-output helper -- so dummy values are sufficient.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")
