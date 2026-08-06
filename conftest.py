"""Pytest bootstrap for the evals package.

`evals.utils` instantiates the OpenAI client at import time and reads
``OPENAI_API_KEY`` / ``OPENAI_ORGANIZATION_ID`` from the environment. Tests in
this repo never hit the network (they stub the summary call), so we provide
non-empty placeholder values here before any ``evals.*`` module is imported.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-placeholder-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-placeholder-org")
