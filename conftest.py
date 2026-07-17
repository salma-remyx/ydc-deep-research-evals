"""Pytest bootstrap for the evals test suite.

``evals.utils`` constructs the OpenAI client at import time from
``OPENAI_API_KEY`` / ``OPENAI_ORGANIZATION_ID``. Provide dummies during
collection so importing ``evals.*`` succeeds; tests mock the actual
network calls and never touch a real account.

A conftest at the repository root also makes the repo root importable as a
package source (so ``import evals...`` resolves under pytest).
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")
