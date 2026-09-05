"""Every provider on our list silently ignores parameters it does not recognise.

Anthropic's compatibility layer drops `seed`, `response_format` and
`reasoning_effort` without a word and caps `temperature` at 1; Google's says the
same in its own documentation. So what a run actually asked for is knowable only
from what the harness wrote down.
"""
from __future__ import annotations

import pytest
from conftest import completion

from evals.client import ChatClient, Endpoint, reasoning_of


def test_an_unknown_sampling_key_is_refused_rather_than_forwarded():
    """A sweep was once scored under a penalty that never reached the server."""
    with pytest.raises(ValueError) as e:
        Endpoint(model="m", sampling={"repetition_penalty": 1.1}).check()
    assert "repetition_penalty" in str(e.value)


def test_the_request_body_is_recorded_and_the_key_is_not(provider):
    p = provider(lambda body: completion("hello"))
    endpoint = Endpoint(model="m", base_url=p.url, api_key_env="A_SECRET_VAR",
                        sampling={"temperature": 0.0, "max_tokens": 32},
                        extra_body={"reasoning": {"effort": "high"}}, retries=0)
    attempt = ChatClient(endpoint).complete(None, "a question")

    assert attempt.request["temperature"] == 0.0
    assert attempt.request["max_tokens"] == 32
    assert attempt.request["extra_body"] == {"reasoning": {"effort": "high"}}
    # The prompt is stored once per row, not repeated in every request record.
    assert "messages" not in attempt.request
    assert attempt.request["n_messages"] == 1

    sent = p.requests[0]["body"]
    assert sent["reasoning"] == {"effort": "high"}, "extra_body did not reach the wire"

    # The manifest records the name of the variable, never its value.
    recorded = endpoint.redacted()
    assert recorded["api_key_env"] == "A_SECRET_VAR"
    assert "api_key" not in recorded


@pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
def test_the_reasoning_trace_is_read_under_either_name(provider, key):
    """vLLM renamed one to the other and warns a client can read the empty one."""
    p = provider(lambda body: completion("answer", reasoning_key=key,
                                         reasoning="the thinking"))
    attempt = ChatClient(Endpoint(model="m", base_url=p.url, retries=0)).complete(
        None, "q")
    assert attempt.reasoning == "the thinking"


def test_reasoning_of_reads_a_plain_dict_too():
    assert reasoning_of({"reasoning": "a"}) == "a"
    assert reasoning_of({"reasoning_content": "b"}) == "b"
    assert reasoning_of({}) == ""


def test_a_system_message_is_sent_only_when_there_is_one(provider):
    p = provider(lambda body: completion("x"))
    client = ChatClient(Endpoint(model="m", base_url=p.url, retries=0))
    client.complete(None, "q")
    client.complete("be brief", "q")
    assert [m["role"] for m in p.requests[0]["body"]["messages"]] == ["user"]
    assert [m["role"] for m in p.requests[1]["body"]["messages"]] == ["system", "user"]
