from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, default=str)


class Store:
    """Single-process DuckDB store; callers share one locked connection."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._connection = duckdb.connect(str(path))
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize(self) -> None:
        statements = [
            """CREATE TABLE IF NOT EXISTS career_profiles (
                profile_version INTEGER PRIMARY KEY,
                profile_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS companies (
                company_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS jobs (
                job_id VARCHAR PRIMARY KEY,
                company_id VARCHAR,
                company VARCHAR,
                title VARCHAR,
                source_language VARCHAR,
                country VARCHAR,
                city VARCHAR,
                work_mode VARCHAR,
                seniority VARCHAR,
                employment_type VARCHAR,
                salary_min DOUBLE,
                salary_max DOUBLE,
                salary_currency VARCHAR,
                required_experience_years DOUBLE,
                source_url VARCHAR,
                reference_id VARCHAR,
                original_text TEXT NOT NULL,
                normalized_description TEXT,
                english_summary TEXT,
                extraction_json JSON NOT NULL,
                application_deadline DATE,
                import_run_id VARCHAR,
                imported_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS job_sources (
                source_id VARCHAR PRIMARY KEY,
                job_id VARCHAR,
                source_path VARCHAR NOT NULL,
                source_url VARCHAR,
                exact_hash VARCHAR NOT NULL UNIQUE,
                normalized_hash VARCHAR NOT NULL,
                duplicate_status VARCHAR NOT NULL,
                potential_duplicate_job_id VARCHAR,
                source_text TEXT NOT NULL,
                source_modified_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL,
                processing_state VARCHAR NOT NULL DEFAULT 'pending',
                failure_stage VARCHAR,
                error_message TEXT,
                configuration_hash VARCHAR,
                extraction_hash VARCHAR
            )""",
            """CREATE TABLE IF NOT EXISTS job_skills (
                job_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                canonical_id VARCHAR,
                requirement VARCHAR NOT NULL,
                evidence TEXT,
                PRIMARY KEY (job_id, name, requirement)
            )""",
            """CREATE TABLE IF NOT EXISTS company_research (
                company_id VARCHAR PRIMARY KEY,
                research_json JSON NOT NULL,
                researched_at TIMESTAMP NOT NULL,
                configuration_hash VARCHAR NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS research_sources (
                source_id VARCHAR PRIMARY KEY,
                company_id VARCHAR NOT NULL,
                url VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                retrieved_at TIMESTAMP NOT NULL,
                UNIQUE(company_id, url)
            )""",
            """CREATE TABLE IF NOT EXISTS job_evaluations (
                evaluation_id VARCHAR PRIMARY KEY,
                job_id VARCHAR NOT NULL,
                profile_version INTEGER NOT NULL,
                evaluation_json JSON NOT NULL,
                weighted_score DOUBLE NOT NULL,
                created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS applications (
                job_id VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS application_events (
                event_id VARCHAR PRIMARY KEY,
                job_id VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                event_date TIMESTAMP NOT NULL,
                feedback TEXT,
                notes TEXT,
                created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS processing_runs (
                run_id VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                input_directory VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                discovered INTEGER NOT NULL DEFAULT 0,
                new_jobs INTEGER NOT NULL DEFAULT 0,
                updated_jobs INTEGER NOT NULL DEFAULT 0,
                unchanged_jobs INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                details_json JSON NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS processing_errors (
                error_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                source_path VARCHAR,
                stage VARCHAR NOT NULL,
                message TEXT NOT NULL,
                retryable BOOLEAN NOT NULL,
                created_at TIMESTAMP NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS analysis_reports (
                report_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                report_type VARCHAR NOT NULL,
                sample_size INTEGER NOT NULL,
                profile_version INTEGER NOT NULL,
                limitations_json JSON NOT NULL,
                report_json JSON NOT NULL,
                created_at TIMESTAMP NOT NULL
            )""",
        ]
        with self._lock:
            for statement in statements:
                self._connection.execute(statement)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            self._connection.execute("BEGIN TRANSACTION")
            try:
                yield
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            result = self._connection.execute(sql, params or [])
            columns = [column[0] for column in result.description or []]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def _execute(self, sql: str, params: list[Any] | None = None) -> None:
        with self._lock:
            self._connection.execute(sql, params or [])

    def save_profile(self, profile: Any) -> int:
        with self.transaction():
            row = self._connection.execute(
                "SELECT COALESCE(MAX(profile_version), 0) + 1 FROM career_profiles"
            ).fetchone()
            version = int(row[0])
            self._connection.execute(
                "INSERT INTO career_profiles VALUES (?, ?, ?)",
                [version, _json(profile), _now()],
            )
        return version

    def latest_profile(self) -> tuple[int, dict[str, Any]] | None:
        rows = self._query(
            "SELECT profile_version, profile_json FROM career_profiles "
            "ORDER BY profile_version DESC LIMIT 1"
        )
        if not rows:
            return None
        value = rows[0]["profile_json"]
        return int(rows[0]["profile_version"]), (
            json.loads(value) if isinstance(value, str) else value
        )

    def create_run(self, input_directory: str) -> str:
        run_id = str(uuid.uuid4())
        self._execute(
            "INSERT INTO processing_runs "
            "(run_id,status,input_directory,started_at,details_json) VALUES (?, ?, ?, ?, ?)",
            [run_id, "running", input_directory, _now(), "{}"],
        )
        return run_id

    def recover_interrupted_runs(self) -> None:
        now = _now()
        with self.transaction():
            self._connection.execute(
                """UPDATE processing_runs SET status='interrupted', completed_at=?,
                   details_json=? WHERE status='running'""",
                [now, _json({"message": "Backend stopped before the run completed"})],
            )
            self._connection.execute(
                """UPDATE job_sources SET processing_state='failed',
                   failure_stage='restart',
                   error_message='Backend stopped before source processing completed'
                   WHERE processing_state='in_progress'"""
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        counts: dict[str, int],
        details: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            """UPDATE processing_runs SET status=?, completed_at=?, discovered=?,
               new_jobs=?, updated_jobs=?, unchanged_jobs=?, error_count=?, details_json=?
               WHERE run_id=?""",
            [
                status,
                _now(),
                counts.get("discovered", 0),
                counts.get("new_jobs", 0),
                counts.get("updated_jobs", 0),
                counts.get("unchanged_jobs", 0),
                counts.get("errors", 0),
                _json(details or {}),
                run_id,
            ],
        )

    def add_error(
        self, run_id: str, source_path: str | None, stage: str, message: str, retryable: bool
    ) -> None:
        self._execute(
            "INSERT INTO processing_errors VALUES (?, ?, ?, ?, ?, ?, ?)",
            [str(uuid.uuid4()), run_id, source_path, stage, message, retryable, _now()],
        )

    def find_source(self, exact_hash: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM job_sources WHERE exact_hash=?", [exact_hash]
        )
        return rows[0] if rows else None

    def find_normalized_source(self, normalized_hash: str) -> dict[str, Any] | None:
        rows = self._query(
            """SELECT * FROM job_sources
               WHERE normalized_hash=? AND job_id IS NOT NULL
               ORDER BY CASE WHEN processing_state='completed' THEN 0 ELSE 1 END,
                        created_at DESC
               LIMIT 1""",
            [normalized_hash],
        )
        return rows[0] if rows else None

    def source_by_path(self, source_path: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM job_sources WHERE source_path=? ORDER BY created_at DESC LIMIT 1",
            [source_path],
        )
        return rows[0] if rows else None

    def find_identity_candidates(self, company: str, title: str) -> list[dict[str, Any]]:
        return self._query(
            """SELECT job_id, company, title, city, country, reference_id, original_text
               FROM jobs WHERE lower(coalesce(company,''))=lower(?)
               AND lower(coalesce(title,''))=lower(?)""",
            [company, title],
        )

    def create_job(
        self,
        extraction: Any,
        original_text: str,
        run_id: str,
    ) -> str:
        job_id = str(uuid.uuid4())
        data = extraction.model_dump(mode="json")
        company = data.get("company")
        company_id = self._ensure_company(company)
        now = _now()
        with self.transaction():
            self._connection.execute(
                """INSERT INTO jobs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                [
                    job_id,
                    company_id,
                    company,
                    data.get("title"),
                    data.get("source_language"),
                    data.get("country"),
                    data.get("city"),
                    data.get("work_mode", "unknown"),
                    data.get("seniority"),
                    data.get("employment_type"),
                    data.get("salary_min"),
                    data.get("salary_max"),
                    data.get("salary_currency"),
                    data.get("required_experience_years"),
                    data.get("source_url"),
                    data.get("reference_id"),
                    original_text,
                    data.get("normalized_description"),
                    data.get("english_summary"),
                    _json(extraction),
                    data.get("application_deadline"),
                    run_id,
                    now,
                    now,
                ],
            )
            self._insert_skills(job_id, data.get("skills", []))
            self._ensure_application(job_id)
        return job_id

    def update_job_analysis(self, job_id: str, extraction: Any, original_text: str, run_id: str) -> None:
        data = extraction.model_dump(mode="json")
        company_id = self._ensure_company(data.get("company"))
        with self.transaction():
            self._connection.execute(
                """UPDATE jobs SET company_id=?, company=?, title=?, source_language=?, country=?, city=?,
                   work_mode=?, seniority=?, employment_type=?, salary_min=?, salary_max=?,
                   salary_currency=?, required_experience_years=?, source_url=?, reference_id=?,
                   original_text=?, normalized_description=?, english_summary=?, extraction_json=?,
                   application_deadline=?, import_run_id=?, updated_at=? WHERE job_id=?""",
                [
                    company_id,
                    data.get("company"),
                    data.get("title"),
                    data.get("source_language"),
                    data.get("country"),
                    data.get("city"),
                    data.get("work_mode", "unknown"),
                    data.get("seniority"),
                    data.get("employment_type"),
                    data.get("salary_min"),
                    data.get("salary_max"),
                    data.get("salary_currency"),
                    data.get("required_experience_years"),
                    data.get("source_url"),
                    data.get("reference_id"),
                    original_text,
                    data.get("normalized_description"),
                    data.get("english_summary"),
                    _json(extraction),
                    data.get("application_deadline"),
                    run_id,
                    _now(),
                    job_id,
                ],
            )
            self._connection.execute("DELETE FROM job_skills WHERE job_id=?", [job_id])
            self._insert_skills(job_id, data.get("skills", []))

    def add_source(
        self,
        job_id: str | None,
        source_path: str,
        source_url: str | None,
        exact_hash: str,
        normalized_hash: str,
        duplicate_status: str,
        source_text: str,
        modified_at: datetime | None,
        potential_duplicate_job_id: str | None = None,
        processing_state: str = "in_progress",
        failure_stage: str | None = None,
        error_message: str | None = None,
        configuration_hash: str | None = None,
        extraction_hash: str | None = None,
    ) -> None:
        self._execute(
            """INSERT INTO job_sources
               (source_id,job_id,source_path,source_url,exact_hash,normalized_hash,
                duplicate_status,potential_duplicate_job_id,source_text,source_modified_at,
                created_at,processing_state,failure_stage,error_message,configuration_hash,
                extraction_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                str(uuid.uuid4()),
                job_id,
                source_path,
                source_url,
                exact_hash,
                normalized_hash,
                duplicate_status,
                potential_duplicate_job_id,
                source_text,
                modified_at,
                _now(),
                processing_state,
                failure_stage,
                error_message,
                configuration_hash,
                extraction_hash,
            ],
        )

    def update_source(
        self,
        exact_hash: str,
        *,
        job_id: str | None = None,
        duplicate_status: str | None = None,
        potential_duplicate_job_id: str | None = None,
        processing_state: str,
        failure_stage: str | None = None,
        error_message: str | None = None,
        configuration_hash: str | None = None,
        extraction_hash: str | None = None,
    ) -> None:
        self._execute(
            """UPDATE job_sources SET job_id=coalesce(?,job_id),
               duplicate_status=coalesce(?,duplicate_status),
               potential_duplicate_job_id=coalesce(?,potential_duplicate_job_id),
               processing_state=?, failure_stage=?, error_message=?,
               configuration_hash=coalesce(?,configuration_hash),
               extraction_hash=coalesce(?,extraction_hash) WHERE exact_hash=?""",
            [
                job_id, duplicate_status, potential_duplicate_job_id, processing_state,
                failure_stage, error_message, configuration_hash, extraction_hash, exact_hash,
            ],
        )

    def _ensure_company(self, name: str | None) -> str | None:
        if not name:
            return None
        normalized = name.casefold().strip()
        company_id = str(uuid.uuid5(uuid.NAMESPACE_URL, normalized))
        with self._lock:
            row = self._connection.execute(
                "SELECT company_id FROM companies WHERE lower(name)=?", [normalized]
            ).fetchone()
            if row:
                return row[0]
            self._connection.execute(
                "INSERT INTO companies VALUES (?, ?, ?)",
                [company_id, name.strip(), _now()],
            )
        return company_id

    def _insert_skills(self, job_id: str, skills: list[dict[str, Any]]) -> None:
        for skill in skills:
            self._connection.execute(
                "INSERT INTO job_skills VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [
                    job_id,
                    skill.get("name", ""),
                    skill.get("canonical_id"),
                    skill.get("requirement", "mandatory"),
                    skill.get("evidence"),
                ],
            )

    def _ensure_application(self, job_id: str) -> None:
        self._connection.execute(
            "INSERT INTO applications VALUES (?, ?, ?, ?) ON CONFLICT(job_id) DO NOTHING",
            [job_id, "not_applied", "", _now()],
        )

    def save_research(
        self, company: str, research: Any, configuration_hash: str
    ) -> str:
        company_id = self._ensure_company(company)
        if company_id is None:
            raise ValueError("Company name is required for research persistence")
        data = research.model_dump(mode="json")
        with self.transaction():
            self._connection.execute(
                """INSERT INTO company_research VALUES (?, ?, ?, ?)
                   ON CONFLICT(company_id) DO UPDATE SET
                   research_json=excluded.research_json, researched_at=excluded.researched_at,
                   configuration_hash=excluded.configuration_hash""",
                [company_id, _json(research), _now(), configuration_hash],
            )
            for source in data.get("sources", []):
                self._connection.execute(
                    """INSERT INTO research_sources VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(company_id, url) DO UPDATE SET
                       title=excluded.title, category=excluded.category,
                       retrieved_at=excluded.retrieved_at""",
                    [
                        str(uuid.uuid4()),
                        company_id,
                        str(source["url"]),
                        source["title"],
                        source.get("category", "general"),
                        source["retrieved_at"],
                    ],
                )
        return company_id

    def get_research(
        self, company: str, configuration_hash: str | None = None
    ) -> dict[str, Any] | None:
        rows = self._query(
            """SELECT cr.research_json, cr.configuration_hash FROM company_research cr JOIN companies c
               ON cr.company_id=c.company_id WHERE lower(c.name)=lower(?)
               ORDER BY cr.researched_at DESC LIMIT 1""",
            [company],
        )
        if configuration_hash and rows and rows[0]["configuration_hash"] != configuration_hash:
            return None
        return self._decode_json(rows[0]["research_json"]) if rows else None

    def save_evaluation(
        self, job_id: str, profile_version: int, evaluation: Any, weighted_score: float
    ) -> str:
        evaluation_id = str(uuid.uuid4())
        self._execute(
            "INSERT INTO job_evaluations VALUES (?, ?, ?, ?, ?, ?)",
            [evaluation_id, job_id, profile_version, _json(evaluation), weighted_score, _now()],
        )
        return evaluation_id

    def save_report(
        self, run_id: str, report_type: str, sample_size: int, profile_version: int,
        limitations: list[str], report: dict[str, Any]
    ) -> str:
        report_id = str(uuid.uuid4())
        self._execute(
            "INSERT INTO analysis_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                report_id, run_id, report_type, sample_size, profile_version,
                _json(limitations), _json(report), _now(),
            ],
        )
        return report_id

    def save_reports(self, reports: list[dict[str, Any]]) -> list[str]:
        identifiers: list[str] = []
        with self.transaction():
            for report in reports:
                identifiers.append(
                    self.save_report(
                        report["run_id"],
                        report["report_type"],
                        report["sample_size"],
                        report["profile_version"],
                        report["limitations"],
                        report["report_json"],
                    )
                )
        return identifiers

    def list_jobs(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self._query(
            """SELECT j.job_id, j.company, j.title, j.city, j.country, j.work_mode,
               j.imported_at, a.status AS application_status, e.weighted_score,
               e.evaluation_json
               FROM jobs j LEFT JOIN applications a ON a.job_id=j.job_id
               LEFT JOIN job_evaluations e ON e.evaluation_id=(
                   SELECT evaluation_id FROM job_evaluations e2
                   WHERE e2.job_id=j.job_id ORDER BY e2.created_at DESC LIMIT 1
               )
               ORDER BY j.imported_at DESC LIMIT ? OFFSET ?""",
            [limit, offset],
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        rows = self._query(
            """SELECT j.*, a.status AS application_status, a.notes AS application_notes
               FROM jobs j LEFT JOIN applications a ON a.job_id=j.job_id
               WHERE j.job_id=?""",
            [job_id],
        )
        if not rows:
            return None
        job = rows[0]
        job["extraction_json"] = self._decode_json(job["extraction_json"])
        job["evaluation"] = self.latest_evaluation(job_id)
        job["research"] = self.get_research(job["company"]) if job.get("company") else None
        job["application_events"] = self.application_events(job_id)
        job["sources"] = self._query(
            """SELECT source_path, source_url, duplicate_status, potential_duplicate_job_id,
                      source_text, created_at
               FROM job_sources WHERE job_id=? ORDER BY created_at""",
            [job_id],
        )
        return job

    def latest_evaluation(self, job_id: str) -> dict[str, Any] | None:
        rows = self._query(
            """SELECT evaluation_json, weighted_score, profile_version, created_at
               FROM job_evaluations WHERE job_id=? ORDER BY created_at DESC LIMIT 1""",
            [job_id],
        )
        if not rows:
            return None
        row = rows[0]
        row["evaluation_json"] = self._decode_json(row["evaluation_json"])
        return row

    def list_reports(self) -> list[dict[str, Any]]:
        rows = self._query(
            """SELECT report_id, run_id, report_type, sample_size, profile_version, created_at
               FROM analysis_reports ORDER BY created_at DESC"""
        )
        return rows

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM analysis_reports WHERE report_id=?", [report_id])
        if not rows:
            return None
        report = rows[0]
        report["report_json"] = self._decode_json(report["report_json"])
        report["limitations_json"] = self._decode_json(report["limitations_json"])
        return report

    def run_history(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM processing_runs ORDER BY started_at DESC LIMIT 100")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM processing_runs WHERE run_id=?", [run_id])
        if not rows:
            return None
        run = rows[0]
        run["details_json"] = self._decode_json(run["details_json"])
        run["errors"] = self._query(
            "SELECT * FROM processing_errors WHERE run_id=? ORDER BY created_at", [run_id]
        )
        return run

    def analytics_snapshot(self) -> list[dict[str, Any]]:
        rows = self._query(
            """SELECT j.job_id, j.company, j.title, j.city, j.country, j.work_mode,
               j.imported_at, j.import_run_id, j.extraction_json,
               a.status AS application_status, e.evaluation_json, e.weighted_score
               FROM jobs j LEFT JOIN applications a ON a.job_id=j.job_id
               LEFT JOIN job_evaluations e ON e.evaluation_id=(
                   SELECT evaluation_id FROM job_evaluations e2
                   WHERE e2.job_id=j.job_id ORDER BY e2.created_at DESC LIMIT 1
               ) ORDER BY j.imported_at"""
        )
        for row in rows:
            row["extraction_json"] = self._decode_json(row["extraction_json"])
            if row["evaluation_json"] is not None:
                row["evaluation_json"] = self._decode_json(row["evaluation_json"])
        return rows

    def backup(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        escaped = destination.as_posix().replace("'", "''")
        with self._lock:
            self._connection.execute(f"EXPORT DATABASE '{escaped}'")

    def restore(self, backup_directory: Path) -> None:
        if not backup_directory.is_dir():
            raise FileNotFoundError(f"Backup directory does not exist: {backup_directory}")
        temporary = self.path.with_name(f".{self.path.stem}-{uuid.uuid4().hex}.restore.duckdb")
        safety_copy = self.path.with_name(f"{self.path.stem}-pre-restore-{uuid.uuid4().hex}.duckdb")
        escaped_backup = backup_directory.resolve().as_posix().replace("'", "''")
        connection = duckdb.connect(str(temporary))
        try:
            connection.execute(f"IMPORT DATABASE '{escaped_backup}'")
        except Exception:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        connection.close()
        with self._lock:
            self._connection.close()
            try:
                shutil.copy2(self.path, safety_copy)
                os.replace(temporary, self.path)
            except Exception:
                if not self.path.exists() and safety_copy.exists():
                    shutil.copy2(safety_copy, self.path)
                self._connection = duckdb.connect(str(self.path))
                temporary.unlink(missing_ok=True)
                raise
            self._connection = duckdb.connect(str(self.path))

    def update_application(self, job_id: str, update: dict[str, Any]) -> None:
        if not self._query("SELECT job_id FROM jobs WHERE job_id=?", [job_id]):
            raise KeyError(job_id)
        now = _now()
        with self.transaction():
            self._ensure_application(job_id)
            current = self._connection.execute(
                "SELECT status, notes FROM applications WHERE job_id=?", [job_id]
            ).fetchone()
            status = update.get("status") or current[0]
            notes = update.get("notes") if update.get("notes") is not None else current[1]
            self._connection.execute(
                "UPDATE applications SET status=?, notes=?, updated_at=? WHERE job_id=?",
                [status, notes, now, job_id],
            )
            event_type = update.get("event_type")
            if not event_type and update.get("status") and update["status"] != current[0]:
                event_type = f"status_changed:{update['status']}"
            if event_type:
                self._connection.execute(
                    "INSERT INTO application_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        str(uuid.uuid4()),
                        job_id,
                        event_type,
                        update.get("event_date") or now,
                        update.get("feedback"),
                        update.get("notes"),
                        now,
                    ],
                )

    def application_events(self, job_id: str) -> list[dict[str, Any]]:
        return self._query(
            "SELECT * FROM application_events WHERE job_id=? ORDER BY event_date, created_at",
            [job_id],
        )

    @staticmethod
    def _decode_json(value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value
