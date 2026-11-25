def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_predict_and_history_flow(client):
    client.delete("/api/history")
    r = client.post("/api/predict", json={"subject": "Free prize!!!", "body": "Click here to claim your free gift card now"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("spam", "ham") and data["saved"] is True and data["id"]
    h = client.get("/api/history").json()
    assert h["total"] == 1 and h["items"][0]["subject"] == "Free prize!!!"


def test_validation_errors(client):
    assert client.post("/api/predict", json={"subject": "", "body": "   "}).status_code == 422
    assert client.post("/api/predict", json={"subject": 123, "body": "x"}).status_code == 422
    assert client.post("/api/predict", json={"subject": "x", "body": "a" * 20001}).status_code == 422
    assert client.post("/api/predict", json={"subject": "x" * 301, "body": "ok"}).status_code == 422


def test_do_not_save_option(client):
    client.delete("/api/history")
    r = client.post("/api/predict", json={"subject": "Private note", "body": "secret text", "save_message": False})
    assert r.status_code == 200 and r.json()["saved"] is False
    item = client.get("/api/history").json()["items"][0]
    assert item["subject"] == "(not stored)" and item["body"] == "" and item["content_saved"] is False


def test_history_filter_search_pagination_delete(client):
    client.delete("/api/history")
    for i in range(3):
        client.post("/api/predict", json={"subject": f"Meeting {i}", "body": "Hi Sam, agenda for the project review on Friday."})
    client.post("/api/predict", json={"subject": "WIN a free phone", "body": "Click now, limited time offer, claim your prize"})
    assert client.get("/api/history?page_size=2").json()["pages"] == 2
    assert client.get("/api/history?q=Meeting").json()["total"] == 3
    assert client.get("/api/history?label=bad").status_code == 422
    first = client.get("/api/history").json()["items"][0]["id"]
    assert client.delete(f"/api/history/{first}").status_code == 204
    assert client.delete(f"/api/history/{first}").status_code == 404
    assert client.delete("/api/history").json()["deleted"] == 3


def test_stats_and_model_info(client):
    client.delete("/api/history")
    client.post("/api/predict", json={"subject": "Lunch on Friday?", "body": "Are you free to talk about the project?"})
    s = client.get("/api/stats").json()
    assert s["total_predictions"] == 1 and s["spam_count"] + s["ham_count"] == 1
    assert s["metrics"]["f1"] is not None
    info = client.get("/api/model-info").json()
    assert info["model_available"] and "Logistic Regression" in info["model_type"]


def test_response_has_risk_and_findings(client):
    d = client.post("/api/predict", json={"subject": "WIN FREE CASH!!!", "body": "Act now http://bit.ly/x"}).json()
    assert 0 <= d["risk_score"] <= 100 and d["verdict"] and d["findings"]


def test_auth_required(anon_client):
    assert anon_client.get("/api/history").status_code == 401
    assert anon_client.post("/api/predict", json={"subject": "a", "body": "b"}, headers={"X-API-Key": "bad"}).status_code == 401
    assert anon_client.get("/health").status_code == 200


def test_user_limit_and_isolation(client):
    import pytest
    from app import config
    from app.auth import UserLimitReached, create_user
    from app.database import SessionLocal
    client.post("/api/predict", json={"subject": "mine", "body": "hello team"})
    with SessionLocal() as db:
        _, other_key = create_user(db, "other")
        other = client.get("/api/history", headers={"X-API-Key": other_key}).json()
        assert other["total"] == 0  # other user cannot see my history
        n = 0
        with pytest.raises(UserLimitReached):
            while True:
                create_user(db, f"u{n}"); n += 1
        assert n <= config.MAX_USERS
