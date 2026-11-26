import httpx
import pytest
from fastapi.testclient import TestClient

from app import app
from routing import REVIEW, keyword_router


def test_keyword_tie():
    assert keyword_router("My card was charged twice").category == "billing"
    assert keyword_router("tracking and password").category == REVIEW


@pytest.mark.parametrize("failure", ["timeout", "invalid"])
def test_model_failure_then_reviewer_override(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("TICKET_DB", str(tmp_path / "test.sqlite3"))

    def fake_post(*args, **kwargs):
        if failure == "timeout":
            raise httpx.ReadTimeout("test timeout")
        return httpx.Response(200, request=httpx.Request("POST", args[0]), json={"message": {"content": '{"category":[]}'}})

    monkeypatch.setattr(httpx, "post", fake_post)
    with TestClient(app) as client:
        assert client.post("/tickets", data={"body": "My card was charged"}).status_code == 200
        page = client.get("/").text
        assert "Suggested: Manual review" in page
        assert ("Model timed out" if failure == "timeout" else "Invalid model response") in page
        assert client.post("/tickets/1/review", data={"category": "billing"}).status_code == 200
        assert "<strong>Reviewed:</strong> Billing" in client.get("/").text
        assert client.post("/tickets/1/review", data={"category": "shipping"}).status_code == 409


def test_structured_success(monkeypatch):
    from routing import llm_router

    def fake_post(*args, **kwargs):
        return httpx.Response(200, request=httpx.Request("POST", args[0]),
                              json={"message": {"content": '{"category":"shipping"}'}})

    monkeypatch.setattr(httpx, "post", fake_post)
    assert llm_router("Where is my parcel?").category == "shipping"
