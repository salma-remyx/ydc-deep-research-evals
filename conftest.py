"""Pytest configuration shared across the test suite.

The production package constructs the OpenAI client at import time
(``evals.utils`` reads ``OPENAI_API_KEY`` / ``OPENAI_ORGANIZATION_ID``), so the
tests stub those values before any ``evals.*`` module is imported. The stubs are
only used to satisfy import; no real API call is made by the tests below.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org-not-used")
