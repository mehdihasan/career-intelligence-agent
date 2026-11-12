from __future__ import annotations

import logging
import queue
import threading
import hashlib
import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from career_agent.agents import (
    CVExtractionAgent,
    CompanyResearchAgent,
    JobEvaluationAgent,
    JobExtractionAgent,
)
from career_agent.analytics import ReportGenerator
from career_agent.config import Settings, load_profile
from career_agent.ingestion import discover_job_files, extract_possible_urls, read_job_file
from career_agent.llm import LLMProvider
from career_agent.research import DuckDuckGoResearchTool, ResearchTool
from career_agent.schemas import CareerProfile, JobExtraction, RunSummary
from career_agent.storage import Store

logger = logging.getLogger(__name__)


class StageError(RuntimeError):
    def __init__(self, stage: str, original: Exception):
        super().__init__(str(original))
        self.stage = stage


class AnalysisPipeline:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        llm: LLMProvider,
        research_tool: ResearchTool | None = None,
    ):
        self.settings = settings
        self.store = store
        self.llm = llm
        self.extraction_agent = JobExtractionAgent(llm, settings.model)
        self.cv_agent = CVExtractionAgent(llm, settings.model)
        self.research_agent = CompanyResearchAgent(
            llm,
            research_tool or DuckDuckGoResearchTool(),
            settings.research,
            settings.model,
            store,
        )
        self.evaluation_agent = JobEvaluationAgent(
            llm, settings.scoring_weights, settings.model.max_input_chars
        )
        self.report_generator = ReportGenerator(store, llm)

    def extract_cv(self, text: str) -> Any:
        return self.cv_agent.extract(text)

    def execute(self, run_id: str, input_directory: Path, force: bool = False) -> RunSummary:
        run_started_clock = time.perf_counter()
        started = self.store.get_run(run_id)["started_at"]
        llm_metrics_before = self._llm_metrics()
        counts = {
            "discovered": 0,
            "new_jobs": 0,
            "updated_jobs": 0,
            "unchanged_jobs": 0,
            "errors": 0,
        }
        new_job_ids: list[str] = []
        metrics = {
            "extraction_seconds": 0.0,
            "research_seconds": 0.0,
            "evaluation_seconds": 0.0,
            "jobs_processed": 0,
        }
        latest_profile = self.store.latest_profile()
        if latest_profile:
            profile_version, profile_data = latest_profile
            profile = CareerProfile.model_validate(profile_data)
        else:
            profile = load_profile(self.settings.profile_path)
            profile_version = self.store.save_profile(profile)
        configuration_hash = self._configuration_hash(profile_version)
        self.research_agent.begin_run()
        try:
            paths = discover_job_files(input_directory)
            counts["discovered"] = len(paths)
            for path in paths:
                try:
                    outcome = self._process_file(
                        path, run_id, profile_version, profile, force, configuration_hash,
                        metrics,
                    )
                    counts[outcome["count"]] += 1
                    if outcome.get("new_job_id"):
                        new_job_ids.append(outcome["new_job_id"])
                    counts["errors"] += outcome.get("errors", 0)
                except StageError as error:
                    logger.exception("Stage %s failed for job file %s", error.stage, path)
                    self._record_source_failure(path, error.stage, error)
                    self.store.add_error(run_id, str(path), error.stage, str(error), True)
                    counts["errors"] += 1
                except Exception as error:
                    logger.exception("Failed to process job file %s", path)
                    self._record_source_failure(path, "processing", error)
                    self.store.add_error(run_id, str(path), "processing", str(error), True)
                    counts["errors"] += 1
            try:
                self.report_generator.generate(run_id, new_job_ids, profile_version)
            except Exception as error:
                self.store.add_error(run_id, None, "report_generation", str(error), True)
                raise StageError("report_generation", error) from error
            status = "completed_with_errors" if counts["errors"] else "completed"
            self.store.finish_run(
                run_id,
                status,
                counts,
                self._run_details(
                    profile_version, metrics, run_started_clock, llm_metrics_before
                ),
            )
        except StageError as error:
            logger.exception("Analysis run %s failed during %s", run_id, error.stage)
            counts["errors"] += 1
            self.store.finish_run(
                run_id,
                "failed",
                counts,
                {
                    **self._run_details(
                        profile_version, metrics, run_started_clock, llm_metrics_before
                    ),
                    "failure_stage": error.stage,
                },
            )
        except Exception as error:
            logger.exception("Analysis run %s failed", run_id)
            counts["errors"] += 1
            self.store.add_error(run_id, None, "run", str(error), True)
            self.store.finish_run(
                run_id,
                "failed",
                counts,
                {
                    **self._run_details(
                        profile_version, metrics, run_started_clock, llm_metrics_before
                    ),
                    "failure_stage": "run",
                },
            )
        run = self.store.get_run(run_id)
        return RunSummary(
            run_id=run_id,
            status=run["status"],
            discovered=counts["discovered"],
            new_jobs=counts["new_jobs"],
            updated_jobs=counts["updated_jobs"],
            unchanged_jobs=counts["unchanged_jobs"],
            errors=counts["errors"],
            started_at=started,
            completed_at=run.get("completed_at"),
            details=run.get("details_json") or {},
        )

    def _process_file(
        self,
        path: Path,
        run_id: str,
        profile_version: int,
        profile: CareerProfile,
        force: bool,
        configuration_hash: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            document = read_job_file(path)
        except (OSError, ValueError) as error:
            raise StageError("ingestion", error) from error
        existing_source = self.store.find_source(document.exact_hash)
        if (
            existing_source
            and existing_source["processing_state"] == "completed"
            and existing_source.get("configuration_hash") == configuration_hash
            and not force
        ):
            return {"count": "unchanged_jobs"}

        errors = 0
        job_id = existing_source.get("job_id") if existing_source else None
        duplicate_status = existing_source.get("duplicate_status") if existing_source else "new"
        potential_duplicate = (
            existing_source.get("potential_duplicate_job_id") if existing_source else None
        )
        extraction: JobExtraction | None = None
        count_key: str | None = None
        new_job_id: str | None = None
        if not job_id:
            normalized_source = self.store.find_normalized_source(document.normalized_hash)
            if normalized_source and normalized_source.get("job_id"):
                job_id = normalized_source["job_id"]
                duplicate_status = "normalized_duplicate"
                job = self.store.get_job(job_id)
                extraction = JobExtraction.model_validate(job["extraction_json"])
                if existing_source:
                    self.store.update_source(
                        document.exact_hash,
                        job_id=job_id,
                        duplicate_status=duplicate_status,
                        processing_state="in_progress",
                        configuration_hash=configuration_hash,
                    )
                else:
                    self.store.add_source(
                        job_id,
                        str(path),
                        extraction.source_url,
                        document.exact_hash,
                        document.normalized_hash,
                        duplicate_status,
                        document.content,
                        document.modified_at,
                        processing_state="in_progress",
                        configuration_hash=configuration_hash,
                        extraction_hash=normalized_source.get("extraction_hash"),
                    )
                cached_extraction_valid = (
                    not force
                    and normalized_source.get("extraction_hash") == self._extraction_hash()
                )
                if (
                    cached_extraction_valid
                    and normalized_source.get("processing_state") == "completed"
                    and normalized_source.get("configuration_hash") == configuration_hash
                ):
                    self.store.update_source(
                        document.exact_hash,
                        job_id=job_id,
                        duplicate_status=duplicate_status,
                        processing_state="completed",
                        configuration_hash=configuration_hash,
                    )
                    return {"count": "unchanged_jobs"}
                count_key = "updated_jobs"
                if not cached_extraction_valid:
                    extraction = None
        extraction_cache_valid = bool(
            existing_source
            and job_id
            and existing_source.get("extraction_hash") == self._extraction_hash()
            and not force
        )
        if extraction is None and extraction_cache_valid:
            job = self.store.get_job(job_id)
            extraction = JobExtraction.model_validate(job["extraction_json"])
            count_key = (
                "unchanged_jobs"
                if existing_source.get("configuration_hash") == configuration_hash
                else "updated_jobs"
            )
        if extraction is None:
            extraction_started = time.perf_counter()
            try:
                extraction = self.extraction_agent.extract(
                    document.text, extract_possible_urls(document.content)
                )
            except Exception as error:
                raise StageError("extraction", error) from error
            finally:
                metrics["extraction_seconds"] += time.perf_counter() - extraction_started
            if not job_id:
                job_id, duplicate_status, potential_duplicate = self._resolve_identity(
                    extraction, document.text
                )
                if job_id:
                    self.store.update_job_analysis(
                        job_id, extraction, document.text, run_id
                    )
                    count_key = "updated_jobs"
                else:
                    job_id = self.store.create_job(
                        extraction, document.text, run_id
                    )
                    count_key = "new_jobs"
                    if duplicate_status == "new":
                        new_job_id = job_id
            else:
                self.store.update_job_analysis(job_id, extraction, document.text, run_id)
                count_key = "updated_jobs"
            if existing_source and not existing_source.get("job_id"):
                self.store.update_source(
                    document.exact_hash,
                    job_id=job_id,
                    duplicate_status=duplicate_status,
                    potential_duplicate_job_id=potential_duplicate,
                    processing_state="in_progress",
                    configuration_hash=configuration_hash,
                    extraction_hash=self._extraction_hash(),
                )
            elif not existing_source and not self.store.find_source(document.exact_hash):
                self.store.add_source(
                    job_id,
                    str(path),
                    extraction.source_url,
                    document.exact_hash,
                    document.normalized_hash,
                    duplicate_status,
                    document.content,
                    document.modified_at,
                    potential_duplicate,
                    configuration_hash=configuration_hash,
                    extraction_hash=self._extraction_hash(),
                )
                if duplicate_status == "new":
                    new_job_id = job_id
        if count_key is None:
            count_key = "unchanged_jobs"
        if not job_id:
            raise RuntimeError(f"No canonical job was assigned for source {path}")
        if duplicate_status == "ambiguous":
            self.store.add_error(
                run_id,
                str(path),
                "deduplication",
                f"Possible duplicate of job {potential_duplicate}; manual review required",
                False,
            )
            errors += 1
        research = None
        if extraction.company:
            research_started = time.perf_counter()
            try:
                outcome = self.research_agent.research(extraction.company)
                research = outcome.research
                for message in outcome.errors:
                    self.store.add_error(run_id, str(path), "research", message, True)
                    errors += 1
            except Exception as error:
                logger.exception("Company research failed for %s", extraction.company)
                self.store.add_error(run_id, str(path), "research", str(error), True)
                errors += 1
            finally:
                metrics["research_seconds"] += time.perf_counter() - research_started
        evaluation_started = time.perf_counter()
        try:
            evaluation = self.evaluation_agent.evaluate(extraction, profile, research)
        except Exception as error:
            raise StageError("evaluation", error) from error
        metrics["evaluation_seconds"] += time.perf_counter() - evaluation_started
        metrics["jobs_processed"] += 1
        self.store.save_evaluation(
            job_id,
            profile_version,
            evaluation,
            evaluation.overall_opportunity_score,
        )
        self.store.update_source(
            document.exact_hash,
            job_id=job_id,
            duplicate_status=duplicate_status,
            potential_duplicate_job_id=potential_duplicate,
            processing_state="completed",
            configuration_hash=configuration_hash,
            extraction_hash=self._extraction_hash(),
        )
        result = {"count": count_key, "errors": errors}
        if count_key == "new_jobs":
            result["new_job_id"] = job_id
        return result

    def _resolve_identity(
        self, extraction: JobExtraction, text: str
    ) -> tuple[str | None, str, str | None]:
        if not extraction.company or not extraction.title:
            return None, "new", None
        candidates = self.store.find_identity_candidates(
            extraction.company.strip(), extraction.title.strip()
        )
        normalized = " ".join(text.casefold().split())
        best: tuple[float, dict[str, Any]] | None = None
        for candidate in candidates:
            if (
                extraction.reference_id
                and candidate.get("reference_id")
                and extraction.reference_id.casefold().strip()
                != candidate["reference_id"].casefold().strip()
            ):
                continue
            if _location_conflict(
                extraction.city,
                extraction.country,
                candidate.get("city"),
                candidate.get("country"),
            ):
                continue
            old_text = " ".join(candidate["original_text"].casefold().split())
            score = SequenceMatcher(None, normalized, old_text, autojunk=False).ratio()
            if best is None or score > best[0]:
                best = (score, candidate)
        same_reference = bool(
            best
            and extraction.reference_id
            and best[1].get("reference_id")
            and extraction.reference_id.casefold().strip()
            == best[1]["reference_id"].casefold().strip()
        )
        if best and (best[0] >= 0.97 or same_reference):
            return best[1]["job_id"], "updated_version", None
        if best and best[0] >= 0.5:
            return None, "ambiguous", best[1]["job_id"]
        return None, "new", None

    def _record_source_failure(self, path: Path, stage: str, error: Exception) -> None:
        try:
            document = read_job_file(path)
        except (OSError, ValueError):
            return
        source = self.store.find_source(document.exact_hash)
        if source:
            self.store.update_source(
                document.exact_hash,
                processing_state="failed",
                failure_stage=stage,
                error_message=str(error),
            )
        else:
            self.store.add_source(
                None,
                str(path),
                None,
                document.exact_hash,
                document.normalized_hash,
                "failed",
                document.content,
                document.modified_at,
                processing_state="failed",
                failure_stage=stage,
                error_message=str(error),
            )

    def _configuration_hash(self, profile_version: int) -> str:
        relevant = {
            "profile_version": profile_version,
            "model": self.settings.model.model_dump(mode="json"),
            "research": self.settings.research.model_dump(mode="json"),
            "scoring_weights": self.settings.scoring_weights.model_dump(mode="json"),
        }
        return hashlib.sha256(
            json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _extraction_hash(self) -> str:
        relevant = self.settings.model.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _run_details(
        self,
        profile_version: int,
        metrics: dict[str, Any],
        run_started_clock: float,
        llm_metrics_before: dict[str, Any],
    ) -> dict[str, Any]:
        current_llm_metrics = self._llm_metrics()
        run_llm_metrics = {
            key: (
                round(value - llm_metrics_before.get(key, 0), 3)
                if isinstance(value, float)
                else value - llm_metrics_before.get(key, 0)
            )
            for key, value in current_llm_metrics.items()
            if key in {"generation_calls", "generated_tokens", "generation_seconds"}
        }
        generation_seconds = run_llm_metrics.get("generation_seconds", 0)
        run_llm_metrics["tokens_per_second"] = (
            round(run_llm_metrics.get("generated_tokens", 0) / generation_seconds, 2)
            if generation_seconds
            else None
        )
        for key in ("mps_peak_allocated_memory_bytes", "process_peak_rss_bytes"):
            run_llm_metrics[key] = current_llm_metrics.get(key)
        return {
            "profile_version": profile_version,
            "metrics": {
                **{key: round(value, 3) if isinstance(value, float) else value for key, value in metrics.items()},
                "total_seconds": round(time.perf_counter() - run_started_clock, 3),
                "llm": run_llm_metrics,
            },
        }

    def _llm_metrics(self) -> dict[str, Any]:
        metrics = getattr(self.llm, "metrics", None)
        return metrics() if callable(metrics) else {}


def _location_conflict(
    incoming_city: str | None,
    incoming_country: str | None,
    existing_city: str | None,
    existing_country: str | None,
) -> bool:
    if incoming_city and existing_city and incoming_city.casefold().strip() != existing_city.casefold().strip():
        return True
    aliases = {
        "us": "united states",
        "u.s.": "united states",
        "united states of america": "united states",
        "sverige": "sweden",
        "বাংলাদেশ": "bangladesh",
    }
    if incoming_country and existing_country:
        incoming = aliases.get(incoming_country.casefold().strip(), incoming_country.casefold().strip())
        existing = aliases.get(existing_country.casefold().strip(), existing_country.casefold().strip())
        return incoming != existing
    return False


class RunQueue:
    """A single background worker that rejects overlapping analysis runs."""

    def __init__(self, pipeline: AnalysisPipeline, store: Store, default_input: Path):
        self.pipeline = pipeline
        self.store = store
        self.default_input = default_input
        self._queue: queue.Queue[tuple[str, Path, bool] | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._active_run_id: str | None = None
        self._worker = threading.Thread(target=self._work, name="career-analysis", daemon=True)
        self._worker.start()

    def start(self, input_directory: Path | None = None, force: bool = False) -> str:
        with self._lock:
            if self._active_run_id is not None:
                raise RuntimeError(f"Analysis run {self._active_run_id} is already active")
            directory = input_directory or self.default_input
            run_id = self.store.create_run(str(directory))
            self._active_run_id = run_id
            self._queue.put_nowait((run_id, directory, force))
            return run_id

    def active_run_id(self) -> str | None:
        with self._lock:
            return self._active_run_id

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            run_id, directory, force = item
            try:
                self.pipeline.execute(run_id, directory, force)
            except Exception as error:
                logger.exception("Unhandled analysis queue failure for run %s", run_id)
                self.store.add_error(run_id, None, "queue", str(error), True)
                self.store.finish_run(run_id, "failed", {"errors": 1})
            finally:
                with self._lock:
                    self._active_run_id = None
                self._queue.task_done()

    def stop(self) -> None:
        self._queue.put(None)
        self._worker.join()
