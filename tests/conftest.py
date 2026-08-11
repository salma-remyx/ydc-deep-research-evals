"""Shared test setup.

Importing ``evals.utils`` constructs an OpenAI client from environment
variables at module load time. Tests in this suite never call the network --
the LLM-backed methods are stubbed -- but the client object still needs to be
constructable, so provide dummy credentials when none are present.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")
