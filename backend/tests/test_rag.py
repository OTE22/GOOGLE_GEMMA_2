from types import SimpleNamespace
from app.rag import chat


def test_agent_limits_retrieval_and_does_not_invent_media_evidence(monkeypatch):
    class Engine:
        def embed_query(self, query):
            return [1.0] * 768
    class Store:
        calls = 0
        def search(self, *args):
            self.calls += 1
            return [{"id": "audio", "file_name": "safety.wav", "text": "", "rank": 1}]
    def completion(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="workplace safety briefing"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    monkeypatch.setattr("app.rag.OpenAI", lambda **kwargs: client)
    settings = SimpleNamespace(openrouter_api_key="test", openrouter_base_url="https://example.invalid", openrouter_model="test")
    store = Store()
    result = chat("What protective equipment is required?", Engine(), store, settings)
    assert store.calls == 2
    assert result["attempts"] == 2
    assert "cannot establish" in result["answer"]
    assert [s["stage"] for s in result["stages"]].count("Rewrite") == 1
