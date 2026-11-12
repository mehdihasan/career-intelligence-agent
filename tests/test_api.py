from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from career_agent.config import Settings


@pytest.mark.parametrize("client_name", ["cli", "dashboard"])
def test_cv_import_waits_for_inference_without_changing_other_timeouts(
    client_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from career_agent import cli, dashboard

    client = cli._request if client_name == "cli" else dashboard.api
    response = Mock(ok=True, content=b"{}", status_code=200)
    response.json.return_value = {}
    request = Mock(return_value=response)
    monkeypatch.setattr("requests.request", request)

    client("POST", "/profile/cv", json={"filename": "cv.txt", "content_base64": ""})
    assert request.call_args.kwargs["timeout"] == (30, None)

    client("GET", "/profile")
    assert request.call_args.kwargs["timeout"] == 30


def test_local_api_profile_and_run_lifecycle(
    tmp_path: Path,
    monkeypatch,
    fake_llm: object,
    empty_research: object,
) -> None:
    monkeypatch.setenv("CAREER_AGENT_HOME", str(tmp_path / "module-default"))
    from career_agent.api import create_app

    input_directory = tmp_path / "jobs"
    input_directory.mkdir()
    (input_directory / "role.txt").write_text(
        "Acme is hiring a backend engineer. PostgreSQL required.", encoding="utf-8"
    )
    settings = Settings(
        data_dir=tmp_path / "app-data",
        input_directory=input_directory,
    )
    app = create_app(settings, fake_llm, empty_research)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        profile = client.get("/profile").json()
        assert profile["profile_version"] == 1
        response = client.post("/runs", json={"force": False})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        for _ in range(100):
            run = client.get(f"/runs/{run_id}").json()
            if run["status"] != "running":
                break
            time.sleep(0.02)
        assert run["status"] == "completed"
        assert run["new_jobs"] == 1
        assert len(client.get("/reports").json()) == 2
        assert len(client.get("/jobs").json()) == 1
        current_settings = client.get("/settings").json()
        current_settings["input_directory"] = str(tmp_path / "next-jobs")
        updated = client.put(
            "/settings",
            json={
                "model": current_settings["model"],
                "research": current_settings["research"],
                "scoring_weights": current_settings["scoring_weights"],
                "input_directory": current_settings["input_directory"],
            },
        )
        assert updated.status_code == 200
        assert updated.json()["input_directory"] == str(tmp_path / "next-jobs")
