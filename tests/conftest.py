"""Shared pytest configuration.

Importing ``evals.utils`` constructs an ``OpenAI`` client at module load time
and reads ``OPENAI_API_KEY`` / ``OPENAI_ORGANIZATION_ID`` from the environment.
The integration tests never call the API, but the import still needs those
variables to be present. We also put the repo root on ``sys.path`` so the
namespace ``evals`` package is importable regardless of how pytest is invoked.
"""

import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org-not-used")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
