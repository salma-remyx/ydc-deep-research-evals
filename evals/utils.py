import os
import re

from typing import Type, Optional, Literal
from pydantic import BaseModel

from openai import OpenAI
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=os.environ["OPENAI_ORGANIZATION_ID"],
)


def query_openai_model(
    messages: list[ChatCompletionMessageParam],
    model: str = "o3-mini-2025-01-31",
    max_output_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 120,
    response_type: Literal["text", "json_object"] = "text",
) -> dict:
    response_format = {"type": response_type}
    has_no_temperature = model.startswith("o")
    response = client.chat.completions.create(
        model=model,
        temperature=NOT_GIVEN if has_no_temperature else temperature,
        messages=messages,
        max_completion_tokens=max_output_tokens,
        timeout=timeout,
        response_format=response_format,  # type: ignore[call-overload]
    )
    return dict(response.choices[0].message)


def query_openai_model_structured_outputs(
    messages: list[ChatCompletionMessageParam],
    output_class: Type[BaseModel],
    model: str = "o3-mini-2025-01-31",
    max_completion_tokens: int = 5000,
    temperature: float = 0.0,
    timeout: int = 120,
) -> Optional[BaseModel]:
    has_no_temperature = model.startswith("o")
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=output_class,
        max_completion_tokens=max_completion_tokens,
        temperature=NOT_GIVEN if has_no_temperature else temperature,
        timeout=timeout,
    )
    return completion.choices[0].message.parsed


def query_openai_model_logprobs(
    messages: list[ChatCompletionMessageParam],
    model: str = "o3-mini-2025-01-31",
    max_completion_tokens: int = 5,
    temperature: float = 0.0,
    top_logprobs: int = 20,
    timeout: int = 120,
) -> list[dict]:
    """Query the model and return the logprob distribution of the first generated token.

    Unlike ``query_openai_model`` (which returns the decoded message) this exposes the
    per-token logprob distribution at the scoring position, enabling expected-value
    ("continuous") scoring over candidate tokens rather than reading a single decoded
    token. Used by the logprob-verifier metric. Returns a list of
    ``{"token": str, "logprob": float}`` entries for the first generated token, or an
    empty list if the model did not return logprobs (e.g. a model that does not expose
    logprobs). Note: the chosen ``model`` must support ``logprobs``/``top_logprobs``.
    """
    has_no_temperature = model.startswith("o")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=NOT_GIVEN if has_no_temperature else temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
        timeout=timeout,
    )
    content = response.choices[0].logprobs.content if response.choices[0].logprobs else None
    if not content:
        return []
    first_token = content[0]
    return [{"token": tp.token, "logprob": tp.logprob} for tp in first_token.top_logprobs]


def replace_markdown_links_with_text(sentence: str, replacement: str) -> str:
    return re.sub(
        r" ?\(?\[((?:\[)?([^]]+)(?:\])?)\]\(([^)]+)\)\)?", replacement, sentence
    )
