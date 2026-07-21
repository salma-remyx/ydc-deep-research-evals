"""Shared pytest setup.

``evals.utils`` constructs the OpenAI client at import time from environment
variables, so make sure dummy values exist before any test imports it. The
client is never actually called in unit tests -- judge calls are monkeypatched.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-in-unit-tests")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org-not-used-in-unit-tests")
