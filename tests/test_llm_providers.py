

def test_max_tokens_param_is_provider_specific():
    """The output cap must go out under the wire name each server actually reads.

    OpenAI renamed `max_tokens` to `max_completion_tokens`; DeepSeek did not follow and silently
    DROPS the unknown field rather than rejecting it — so sending the wrong name means no cap at
    all, with no error. That regression let 75% of DeepSeek calls overrun their configured limit,
    one reaching 65,536 tokens against a 2,000 cap.
    """
    from src.llm_providers import DeepSeekProvider, OpenAIProvider, OpenRouterProvider

    assert OpenAIProvider.max_tokens_param == "max_completion_tokens"
    assert OpenRouterProvider.max_tokens_param == "max_completion_tokens"
    assert DeepSeekProvider.max_tokens_param == "max_tokens"


def test_deepseek_call_sends_max_tokens(monkeypatch):
    """End-to-end: the key in the outgoing kwargs is `max_tokens` for DeepSeek."""
    from src.llm_providers import DeepSeekProvider

    captured = {}

    class _Completions:
        def create(self, **kw):
            captured.update(kw)
            raise RuntimeError("stop after capturing kwargs")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    prov = DeepSeekProvider()
    monkeypatch.setattr(prov, "_get_client", lambda: _Client())
    try:
        prov.call({"model_id": "deepseek-v4-pro", "system": None, "user": "hi", "max_tokens": 77})
    except RuntimeError:
        pass
    assert captured.get("max_tokens") == 77
    assert "max_completion_tokens" not in captured
