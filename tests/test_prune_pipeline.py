from optimart.prune.pipeline import PruneEngine


def _messages() -> list[dict]:
    docs = (
        "The Nile is the longest river in Africa and flows into the Mediterranean. "
        "Mount Kilimanjaro is a dormant volcano in Tanzania. "
        "Paris is the capital of France and stands on the river Seine. "
        "The Amazon rainforest is located in South America and is extremely biodiverse. "
        "Saturn is a gas giant known for its prominent ring system. "
        "Python is a popular programming language used for data science. "
        "Question: What is the capital of France?"
    )
    return [
        {"role": "system", "content": "You are a careful assistant. Never invent sources."},
        {"role": "user", "content": docs},
    ]


def test_system_and_question_survive():
    engine = PruneEngine(budget_ratio=0.35)
    result = engine.prune_messages(_messages())
    roles = [msg["role"] for msg in result.messages]
    assert "system" in roles
    joined = " ".join(str(msg["content"]) for msg in result.messages)
    assert "Never invent sources" in joined
    assert "capital of France" in joined
    assert result.tokens_after <= result.tokens_before
    assert result.dropped >= 1


def test_query_entity_is_protected():
    engine = PruneEngine(budget_ratio=0.2)
    result = engine.prune_messages(_messages())
    joined = " ".join(str(msg["content"]) for msg in result.messages)
    assert "Paris" in joined


def test_traces_mark_dropped_and_kept():
    engine = PruneEngine(budget_ratio=0.3)
    result = engine.prune_messages(_messages())
    assert result.traces
    kept_text = " ".join(trace.text for trace in result.traces if trace.kept)
    dropped_text = " ".join(trace.text for trace in result.traces if not trace.kept)
    assert "capital of France" in kept_text
    assert dropped_text
    assert any(trace.protected and trace.kept for trace in result.traces)
