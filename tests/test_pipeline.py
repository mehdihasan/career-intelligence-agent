from pathlib import Path

import pytest

from career_agent.config import Settings
from career_agent.pipeline import AnalysisPipeline
from career_agent.schemas import CareerProfile, JobExtraction
from career_agent.storage import Store


def test_run_is_idempotent_and_always_creates_two_reports(
    tmp_path: Path, fake_llm: object, empty_research: object
) -> None:
    input_directory = tmp_path / "jobs"
    input_directory.mkdir()
    content = "<h1>Backend Engineer</h1><p>Acme requires PostgreSQL.</p>"
    (input_directory / "role.html").write_text(content, encoding="utf-8")
    (input_directory / "copy.html").write_text(content, encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "data",
        input_directory=input_directory,
        model={"enabled": True, "max_input_chars": 1000},
    )
    store = Store(settings.database_path)
    try:
        pipeline = AnalysisPipeline(settings, store, fake_llm, empty_research)
        first_id = store.create_run(str(input_directory))
        first = pipeline.execute(first_id, input_directory)
        assert first.status == "completed"
        assert first.new_jobs == 1
        assert first.unchanged_jobs == 1
        assert len(store.list_reports()) == 2

        store.save_profile(CareerProfile(target_roles=["Staff Engineer"]))
        second_id = store.create_run(str(input_directory))
        second = pipeline.execute(second_id, input_directory)
        assert second.status == "completed"
        assert second.new_jobs == 0
        assert second.updated_jobs == 1
        assert second.unchanged_jobs == 1
        reports = store.list_reports()
        assert len(reports) == 4
        new_jobs_report = next(
            store.get_report(item["report_id"])["report_json"]
            for item in reports
            if item["run_id"] == second_id and item["report_type"] == "new_jobs"
        )
        assert new_jobs_report["sample_size"] == 0
        all_jobs_report = next(
            store.get_report(item["report_id"])["report_json"]
            for item in reports
            if item["run_id"] == second_id and item["report_type"] == "all_jobs"
        )
        assert all_jobs_report["sample_size"] == 1
    finally:
        store.close()


def test_identity_matching_respects_location_and_reference_id(
    tmp_path: Path, fake_llm: object, empty_research: object
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    store = Store(settings.database_path)
    try:
        original = JobExtraction(
            company="Acme",
            title="Backend Engineer",
            city="Toronto",
            country="Canada",
            reference_id="posting-1",
        )
        original_id = store.create_job(original, "Backend role at Acme in Toronto.", "run-1")
        pipeline = AnalysisPipeline(settings, store, fake_llm, empty_research)

        different_location = original.model_copy(update={"city": "Vancouver"})
        resolved, status, _ = pipeline._resolve_identity(
            different_location, "Backend role at Acme in Toronto."
        )
        assert (resolved, status) == (None, "new")

        different_posting = original.model_copy(update={"reference_id": "posting-2"})
        resolved, status, _ = pipeline._resolve_identity(
            different_posting, "Backend role at Acme in Toronto."
        )
        assert (resolved, status) == (None, "new")

        same_posting = original.model_copy(update={"normalized_description": "Updated role text"})
        resolved, status, _ = pipeline._resolve_identity(same_posting, "A new shorter posting.")
        assert (resolved, status) == (original_id, "updated_version")
    finally:
        store.close()


def test_normalized_duplicate_is_detected_before_extraction(
    tmp_path: Path,
    fake_llm: object,
    empty_research: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_directory = tmp_path / "jobs"
    input_directory.mkdir()
    first_content = "<h1>Backend Engineer</h1><p>Acme requires PostgreSQL.</p>"
    (input_directory / "first.html").write_text(first_content, encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "data",
        input_directory=input_directory,
        model={"enabled": True, "max_input_chars": 1000},
    )
    store = Store(settings.database_path)
    try:
        pipeline = AnalysisPipeline(settings, store, fake_llm, empty_research)
        first_id = store.create_run(str(input_directory))
        assert pipeline.execute(first_id, input_directory).new_jobs == 1

        duplicate_content = (
            "<div><h1>Backend Engineer</h1></div>"
            "<p>Acme   requires PostgreSQL.</p>"
        )
        (input_directory / "different-markup.html").write_text(
            duplicate_content, encoding="utf-8"
        )

        def unexpected_extraction(*args: object, **kwargs: object) -> JobExtraction:
            raise AssertionError("normalized duplicates must be detected before extraction")

        monkeypatch.setattr(pipeline.extraction_agent, "extract", unexpected_extraction)
        second_id = store.create_run(str(input_directory))
        second = pipeline.execute(second_id, input_directory)
        assert second.status == "completed"
        assert second.new_jobs == 0
        assert second.unchanged_jobs == 2
        assert len(store.list_jobs()) == 1
    finally:
        store.close()
