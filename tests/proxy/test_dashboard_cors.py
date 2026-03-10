"""Tests for CORS headers and query-param token auth on dashboard endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from virtual_context.proxy.metrics import ProxyMetrics
from virtual_context.proxy.dashboard import register_dashboard_routes


@pytest.fixture
def app_with_token():
    app = FastAPI()
    metrics = ProxyMetrics()
    register_dashboard_routes(app, metrics, state=None, dashboard_token="test-token-123")
    return app


@pytest.fixture
def client(app_with_token):
    return TestClient(app_with_token)


def test_cors_headers_on_get(client):
    resp = client.get("/dashboard/settings", headers={"Origin": "https://virtual-context.com"})
    assert resp.headers.get("Access-Control-Allow-Origin") == "https://virtual-context.com"


def test_cors_preflight(client):
    resp = client.options("/dashboard/settings", headers={
        "Origin": "https://virtual-context.com",
        "Access-Control-Request-Method": "PUT",
    })
    assert resp.status_code == 200
    assert "PUT" in resp.headers.get("Access-Control-Allow-Methods", "")


def test_cors_blocked_for_other_origins(client):
    resp = client.get("/dashboard/settings", headers={"Origin": "https://evil.com"})
    assert "Access-Control-Allow-Origin" not in resp.headers


def test_query_param_token_accepted_for_mutating_endpoint(client):
    # compact requires auth — query param should work
    resp = client.post("/dashboard/compact?token=test-token-123")
    assert resp.status_code != 403


def test_query_param_token_rejected_when_wrong(client):
    resp = client.post("/dashboard/compact?token=wrong")
    assert resp.status_code == 403


def test_header_token_still_works(client):
    resp = client.post("/dashboard/compact", headers={"X-VC-Dashboard-Token": "test-token-123"})
    assert resp.status_code != 403
