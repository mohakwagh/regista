import pytest

from regista.errors import ProviderError
from regista.providers import (
    FakeProvider,
    ModelRequest,
    ModelResponse,
    Provider,
    text_response,
    tool_use_response,
)
from regista.types import Message, ToolSpec


def make_request(text: str = "hi") -> ModelRequest:
    return ModelRequest(
        model="fake-model",
        system="be brief",
        messages=[Message.user(text)],
        tools=[ToolSpec(name="t", description="d", input_schema={"type": "object"})],
    )


def test_fake_provider_satisfies_protocol() -> None:
    assert isinstance(FakeProvider([]), Provider)


async def test_fake_provider_serves_script_in_order_and_records_requests() -> None:
    provider = FakeProvider([tool_use_response(("tu_1", "t", {"x": 1})), text_response("done")])
    first = await provider.complete(make_request("start"))
    second = await provider.complete(make_request("next"))

    assert first.stop_reason == "tool_use"
    assert first.message.tool_uses()[0].name == "t"
    assert second.stop_reason == "end_turn"
    assert second.message.text() == "done"
    assert [r.messages[0].text() for r in provider.requests] == ["start", "next"]


async def test_fake_provider_raises_when_exhausted() -> None:
    provider = FakeProvider([])
    with pytest.raises(ProviderError, match="exhausted"):
        await provider.complete(make_request())


def test_request_hash_is_stable_and_sensitive() -> None:
    assert make_request().request_hash() == make_request().request_hash()
    assert make_request("a").request_hash() != make_request("b").request_hash()
    with_params = make_request().model_copy(update={"params": {"temperature": 0.0}})
    assert with_params.request_hash() != make_request().request_hash()


def test_response_trace_dump_excludes_raw_and_round_trips() -> None:
    response = text_response("hello").model_copy(update={"raw": {"secret": "wire-format"}})
    dumped = response.model_dump_trace()
    assert "raw" not in dumped
    restored = ModelResponse.model_validate(dumped)
    assert restored.message.text() == "hello"
    assert restored.raw is None
