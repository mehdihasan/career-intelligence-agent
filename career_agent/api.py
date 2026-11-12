from __future__ import annotations

import base64
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from career_agent.config import (
    ModelSettings,
    ResearchSettings,
    Settings,
    load_profile,
    load_settings,
    save_profile,
    settings_file,
)
from career_agent.ingestion import read_cv
from career_agent.llm import LLMProvider, TransformersProvider
from career_agent.pipeline import AnalysisPipeline, RunQueue
from career_agent.research import ResearchTool
from career_agent.schemas import (
    ApplicationUpdate,
    CareerProfile,
    RunRequest,
    ScoringWeights,
)
from career_agent.storage import Store

logger = logging.getLogger(__name__)


class CVImportRequest(BaseModel):
    filename: str
    content_base64: str


class SettingsUpdate(BaseModel):
    model: ModelSettings
    research: ResearchSettings
    scoring_weights: ScoringWeights
    input_directory: str


class BackupRestoreRequest(BaseModel):
    backup_name: str


def create_app(
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    research_tool: ResearchTool | None = None,
) -> FastAPI:
    config = settings or load_settings()
    store = Store(config.database_path)
    store.recover_interrupted_runs()
    provider = llm or TransformersProvider(config.model)
    pipeline = AnalysisPipeline(config, store, provider, research_tool)
    run_queue = RunQueue(pipeline, store, config.input_directory)
    if store.latest_profile() is None:
        initial = load_profile(config.profile_path)
        store.save_profile(initial)
        save_profile(config.profile_path, initial)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        run_queue.stop()
        store.close()

    app = FastAPI(
        title="Career Intelligence Agent",
        version="0.1.0",
        description="Local-only API for job analysis and career tracking",
        lifespan=lifespan,
    )
    app.state.settings = config
    app.state.store = store
    app.state.provider = provider
    app.state.pipeline = pipeline
    app.state.run_queue = run_queue

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        active = run_queue.active_run_id()
        return {
            "active_run_id": active,
            "job_count": len(store.analytics_snapshot()),
            "database_path": str(config.database_path),
            "model_id": config.model.model_id,
            "model_loaded": bool(getattr(provider, "is_loaded", False)),
            "research_enabled": config.research.enabled,
        }

    @app.get("/profile")
    def get_profile() -> dict[str, Any]:
        latest = store.latest_profile()
        if latest is None:
            return {"profile_version": 0, "profile": CareerProfile().model_dump(mode="json")}
        return {"profile_version": latest[0], "profile": latest[1]}

    @app.put("/profile")
    def put_profile(profile: CareerProfile) -> dict[str, Any]:
        version = store.save_profile(profile)
        save_profile(config.profile_path, profile)
        return {"profile_version": version, "profile": profile.model_dump(mode="json")}

    @app.post("/profile/cv")
    def import_cv(request: CVImportRequest) -> dict[str, Any]:
        suffix = Path(request.filename).suffix.lower()
        if len(request.content_base64) > 15_000_000:
            raise HTTPException(status_code=413, detail="CV upload exceeds 10 MB")
        try:
            content = base64.b64decode(request.content_base64, validate=True)
            if len(content) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="CV upload exceeds 10 MB")
            text = read_cv(Path(request.filename), content)
        except (ValueError, TypeError, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            extracted = pipeline.extract_cv(text)
        except (ValueError, RuntimeError, OSError) as error:
            logger.exception("CV extraction failed for %s", request.filename)
            raise HTTPException(status_code=503, detail=f"CV extraction failed: {error}") from error
        try:
            latest = store.latest_profile()
            profile = CareerProfile.model_validate(latest[1]) if latest else CareerProfile()
            profile.cv_profile = extracted
            profile.cv_filename = Path(request.filename).name
            profile.existing_skills = list(dict.fromkeys([
                *profile.existing_skills,
                *extracted.skills,
                *(technology for experience in extracted.experience for technology in experience.technologies),
            ]))
            profile.updated_at = datetime.now(timezone.utc)
            version = store.save_profile(profile)
            save_profile(config.profile_path, profile)
        except (ValueError, OSError) as error:
            logger.exception("Could not save extracted CV profile for %s", request.filename)
            raise HTTPException(status_code=500, detail=f"Could not save CV profile: {error}") from error
        return {
            "profile_version": version,
            "profile": profile.model_dump(mode="json"),
            "review_required": True,
            "message": "Review and correct the extracted CV information before analysis.",
        }

    @app.get("/settings")
    def get_settings() -> dict[str, Any]:
        return config.model_dump(mode="json")

    @app.put("/settings")
    def put_settings(update: SettingsUpdate) -> dict[str, Any]:
        if run_queue.active_run_id():
            raise HTTPException(status_code=409, detail="Settings cannot change during an analysis run")
        nonlocal config, provider, pipeline
        config.model = update.model
        config.research = update.research
        config.scoring_weights = update.scoring_weights
        config.input_directory = Path(update.input_directory).expanduser()
        latest = store.latest_profile()
        if latest:
            store.save_profile(CareerProfile.model_validate(latest[1]))
        settings_path = settings_file(config)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        provider = TransformersProvider(config.model)
        pipeline = AnalysisPipeline(config, store, provider, research_tool)
        run_queue.pipeline = pipeline
        run_queue.default_input = config.input_directory
        app.state.settings = config
        app.state.provider = provider
        app.state.pipeline = pipeline
        return config.model_dump(mode="json")

    @app.post("/runs", status_code=202)
    def start_run(request: RunRequest) -> dict[str, str]:
        directory = Path(request.input_directory).expanduser() if request.input_directory else None
        try:
            run_id = run_queue.start(directory, request.force)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"run_id": run_id, "status": "running"}

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return store.run_history()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/jobs")
    def list_jobs(
        search: str | None = None,
        role: str | None = None,
        skill: str | None = None,
        location: str | None = None,
        language: str | None = None,
        status_filter: str | None = None,
        minimum_score: float | None = Query(default=None, ge=0, le=100),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        all_rows = store.analytics_snapshot()
        selected: list[dict[str, Any]] = []
        for row in all_rows:
            extracted = row.get("extraction_json") or {}
            evaluation = row.get("evaluation_json") or {}
            text = " ".join(
                str(value or "") for value in (row.get("company"), row.get("title"), row.get("city"), row.get("country"))
            ).casefold()
            if search and search.casefold() not in text:
                continue
            if role and role.casefold() not in str(row.get("title") or "").casefold():
                continue
            if location and location.casefold() not in text:
                continue
            if status_filter and row.get("application_status") != status_filter:
                continue
            if minimum_score is not None and float(row.get("weighted_score") or 0) < minimum_score:
                continue
            if skill and not any(
                skill.casefold() in str(item.get("name", "")).casefold()
                for item in extracted.get("skills", [])
            ):
                continue
            if language and not any(
                language.casefold() in str(item).casefold()
                for item in extracted.get("languages", [])
            ):
                continue
            row["job_fit_score"] = evaluation.get("job_fit_score")
            row["opportunity_score"] = row.get("weighted_score")
            selected.append(row)
        return selected[offset : offset + limit]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.patch("/jobs/{job_id}/application")
    def update_application(job_id: str, update: ApplicationUpdate) -> dict[str, Any]:
        try:
            store.update_application(job_id, update.model_dump(mode="json", exclude_none=True))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        return store.get_job(job_id)

    @app.get("/reports")
    def list_reports() -> list[dict[str, Any]]:
        return store.list_reports()

    @app.get("/reports/{report_id}")
    def get_report(report_id: str) -> dict[str, Any]:
        report = store.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        return report

    @app.post("/backup")
    def create_backup(name: str | None = None) -> dict[str, str]:
        if run_queue.active_run_id():
            raise HTTPException(status_code=409, detail="A backup cannot be created during an analysis run")
        backup_name = name or f"backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if backup_name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", backup_name):
            raise HTTPException(status_code=422, detail="Backup name contains unsupported characters")
        destination = config.data_dir / "backups" / backup_name
        if destination.exists():
            raise HTTPException(status_code=409, detail="A backup with that name already exists")
        store.backup(destination)
        return {"backup_name": backup_name, "path": str(destination)}

    @app.post("/backup/restore")
    def restore_backup(request: BackupRestoreRequest) -> dict[str, str]:
        if run_queue.active_run_id():
            raise HTTPException(status_code=409, detail="A backup cannot be restored during an analysis run")
        if request.backup_name in {".", ".."} or not re.fullmatch(
            r"[A-Za-z0-9._-]{1,80}", request.backup_name
        ):
            raise HTTPException(status_code=422, detail="Backup name contains unsupported characters")
        backup_directory = config.data_dir / "backups" / request.backup_name
        try:
            store.restore(backup_directory)
            store.recover_interrupted_runs()
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            logger.exception("Could not restore database backup %s", request.backup_name)
            raise HTTPException(status_code=500, detail=f"Restore failed: {error}") from error
        return {"backup_name": request.backup_name, "status": "restored"}

    return app


_application: FastAPI | None = None


async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    global _application
    if _application is None:
        _application = create_app()
    await _application(scope, receive, send)
