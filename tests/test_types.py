"""Round-trip and behavior tests for the shared vocabulary."""

import pytest
from pydantic import TypeAdapter, ValidationError

from regista.types import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

BLOCKS: list[ContentBlock] = [
    TextBlock(text="hello"),
    ThinkingBlock(thinking="hmm", signature="sig123"),
    ThinkingBlock(thinking="no signature"),
    ToolUseBlock(id="tu_1", name="read_file", input={"path": "a.txt"}),
    ToolResultBlock(tool_use_id="tu_1", content="contents"),
    ToolResultBlock(tool_use_id="tu_2", content="denied", is_error=True),
]


@pytest.mark.parametrize("block", BLOCKS, ids=lambda b: f"{b.type}")
def test_content_block_round_trip(block: ContentBlock) -> None:
    adapter: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
    assert adapter.validate_json(adapter.dump_json(block)) == block


def test_message_round_trip_preserves_block_order_and_types() -> None:
    message = Message(role="assistant", content=BLOCKS)
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored == message
    assert [type(b) for b in restored.content] == [type(b) for b in BLOCKS]


def test_discriminator_rejects_unknown_block_type() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": [{"type": "banana", "text": "hi"}]})


def test_message_helpers() -> None:
    message = Message(role="assistant", content=BLOCKS)
    assert message.text() == "hello"
    assert [tu.id for tu in message.tool_uses()] == ["tu_1"]
    assert Message.user("hi").role == "user"
    assert Message.assistant("yo").text() == "yo"


def test_usage_addition() -> None:
    a = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=3)
    b = Usage(input_tokens=1, output_tokens=2, cache_write_tokens=4)
    total = a + b
    assert total == Usage(
        input_tokens=11, output_tokens=7, cache_read_tokens=3, cache_write_tokens=4
    )


def test_blocks_are_frozen() -> None:
    with pytest.raises(ValidationError):
        TextBlock(text="x").text = "y"  # type: ignore[misc]
