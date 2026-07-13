from regista.pricing import cost_usd, resolve_price
from regista.types import Usage


def test_resolve_price_supports_popular_ollama_models() -> None:
    assert resolve_price("ollama/llama3") is not None
    assert resolve_price("ollama/mistral") is not None
    assert resolve_price("ollama/qwen2.5:7b") is not None


def test_ollama_models_have_zero_cost() -> None:
    usage = Usage(input_tokens=123, output_tokens=456)
    assert cost_usd("ollama/llama3", usage) == 0.0
    assert cost_usd("ollama/qwen2.5:7b", usage) == 0.0


def test_unknown_models_still_return_none() -> None:
    usage = Usage(input_tokens=10, output_tokens=5)
    assert cost_usd("ollama/unknown-model", usage) is None
