from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
import typer

app = typer.Typer(help="Analyze job opportunities locally and track your career decisions.")
jobs_app = typer.Typer(help="Browse analyzed jobs.")
reports_app = typer.Typer(help="Browse historical reports.")
profile_app = typer.Typer(help="Review and update your career profile.")
app.add_typer(jobs_app, name="jobs")
app.add_typer(reports_app, name="reports")
app.add_typer(profile_app, name="profile")

API_URL = os.environ.get("CAREER_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _request(method: str, path: str, **kwargs: Any) -> Any:
    timeout = (30, None) if method == "POST" and path == "/profile/cv" else 30
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as error:
        typer.echo(
            f"Cannot reach the local API at {API_URL}: {error}\n"
            "Start it with `career-agent serve` in another terminal.",
            err=True,
        )
        raise typer.Exit(1) from error
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        typer.echo(f"API error ({response.status_code}): {detail}", err=True)
        raise typer.Exit(1)
    if not response.content:
        return None
    return response.json()


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@app.command()
def run(
    input_directory: Path | None = typer.Option(
        None, "--input-directory", "-i", help="Directory containing TXT, Markdown, or HTML job ads."
    ),
    force: bool = typer.Option(False, "--force", help="Reprocess unchanged source files."),
    wait: bool = typer.Option(False, "--wait", help="Wait for the run and print its final status."),
) -> None:
    """Start one explicit analysis run."""
    result = _request(
        "POST",
        "/runs",
        json={
            "input_directory": str(input_directory.expanduser()) if input_directory else None,
            "force": force,
        },
    )
    typer.echo(f"Analysis started: {result['run_id']}")
    if wait:
        while True:
            run_status = _request("GET", f"/runs/{result['run_id']}")
            if run_status["status"] != "running":
                _print_json(run_status)
                if run_status["status"] == "failed":
                    raise typer.Exit(1)
                return
            time.sleep(2)


@app.command()
def status() -> None:
    """Show backend and active-run status."""
    _print_json(_request("GET", "/status"))


@jobs_app.command("list")
def list_jobs(
    search: str | None = typer.Option(None, "--search"),
    role: str | None = typer.Option(None, "--role"),
    location: str | None = typer.Option(None, "--location"),
    status_filter: str | None = typer.Option(None, "--status"),
    limit: int = typer.Option(50, min=1, max=1000),
) -> None:
    """List analyzed jobs."""
    params = {
        key: value
        for key, value in {
            "search": search,
            "role": role,
            "location": location,
            "status_filter": status_filter,
            "limit": limit,
        }.items()
        if value is not None
    }
    results = _request("GET", "/jobs", params=params)
    if not results:
        typer.echo("No matching jobs.")
        return
    typer.echo("JOB ID\tCOMPANY\tROLE\tFIT\tOPPORTUNITY\tSTATUS")
    for item in results:
        fit_score = item.get("job_fit_score")
        opportunity_score = item.get("opportunity_score")
        typer.echo(
            f"{item['job_id']}\t{item.get('company') or 'Unknown'}\t"
            f"{item.get('title') or 'Untitled'}\t"
            f"{fit_score if fit_score is not None else '-'}\t"
            f"{opportunity_score if opportunity_score is not None else '-'}\t"
            f"{item.get('application_status') or '-'}"
        )


@jobs_app.command("show")
def show_job(job_id: str) -> None:
    """Show a job advertisement, evaluation, and application history."""
    _print_json(_request("GET", f"/jobs/{job_id}"))


@reports_app.command("list")
def list_reports() -> None:
    """List immutable historical reports."""
    _print_json(_request("GET", "/reports"))


@reports_app.command("show")
def show_report(report_id: str) -> None:
    """Show one report and its calculated statistics."""
    _print_json(_request("GET", f"/reports/{report_id}"))


@profile_app.command("show")
def show_profile() -> None:
    """Show the active career profile."""
    _print_json(_request("GET", "/profile"))


@profile_app.command("import-cv")
def import_cv(path: Path) -> None:
    """Extract CV facts locally for review and correction."""
    file_path = path.expanduser()
    if not file_path.is_file():
        typer.echo(f"CV file does not exist: {file_path}", err=True)
        raise typer.Exit(1)
    content = base64.b64encode(file_path.read_bytes()).decode("ascii")
    typer.echo("Extracting CV locally. This may take several minutes.", err=True)
    result = _request(
        "POST",
        "/profile/cv",
        json={"filename": file_path.name, "content_base64": content},
    )
    _print_json(result)


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="Local API port."),
) -> None:
    """Start the backend bound to loopback only."""
    import uvicorn

    uvicorn.run("career_agent.api:app", host="127.0.0.1", port=port, log_level="info")


@app.command()
def dashboard(port: int = typer.Option(8501, "--port", min=1, max=65535)) -> None:
    """Start the Streamlit dashboard."""
    dashboard_path = Path(__file__).with_name("dashboard.py")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.address=127.0.0.1",
            f"--server.port={port}",
        ],
        check=False,
    )
    if process.returncode:
        raise typer.Exit(process.returncode)


@app.command()
def backup(name: str | None = typer.Option(None, "--name", help="Optional backup name.")) -> None:
    """Export a database backup using the backend-owned connection."""
    _print_json(_request("POST", "/backup", params={"name": name} if name else {}))


@app.command()
def restore(backup_name: str) -> None:
    """Restore a named local database backup."""
    if not typer.confirm(
        f"Replace the active database with backup '{backup_name}'? A safety copy is retained."
    ):
        raise typer.Abort()
    _print_json(_request("POST", "/backup/restore", json={"backup_name": backup_name}))
