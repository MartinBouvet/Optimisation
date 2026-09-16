from fastapi.testclient import TestClient

from optimart.proxy.app import create_app


def test_proxy_dry_run_prunes_and_keeps_system():
    app = create_app({"dry_run": True, "prune": {"budget_ratio": 0.35, "max_query_chars": 80}})
    client = TestClient(app)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a careful assistant. Never invent sources."},
            {
                "role": "user",
                "content": (
                    "The Nile is the longest river in Africa and flows north. "
                    "Kilimanjaro is in Tanzania and attracts climbers. "
                    "Paris is the capital of France and is known for the Louvre. "
                    "Saturn has rings made of ice and rock. "
                    "Python is widely used in machine learning research. "
                    "Question: What is the capital of France?"
                ),
            },
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    body = response.json()
    stats = body["optimart"]
    assert stats["tokens_after"] <= stats["tokens_before"]
    assert stats["dropped"] >= 1
    content = body["choices"][0]["message"]["content"]
    assert "Never invent sources" in content
    assert client.get("/health").json()["status"] == "ok"
