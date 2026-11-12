from pathlib import Path

from career_agent.schemas import JobExtraction, SkillRequirement
from career_agent.storage import Store


def _job() -> JobExtraction:
    return JobExtraction(
        company="Acme",
        title="Backend Engineer",
        normalized_description="Build backend services.",
        skills=[SkillRequirement(name="postgres", requirement="mandatory")],
    )


def test_application_notes_and_event_history_survive_job_updates(tmp_path: Path) -> None:
    store = Store(tmp_path / "career.duckdb")
    try:
        job_id = store.create_job(_job(), "Original advertisement", "run-1")
        store.update_application(
            job_id,
            {
                "status": "technical_interview",
                "notes": "Prepare system design examples.",
                "event_type": "Interview scheduled",
                "feedback": "Panel interview",
            },
        )
        store.update_job_analysis(
            job_id, _job(), "Updated advertisement", "run-2"
        )
        saved = store.get_job(job_id)
        assert saved["application_status"] == "technical_interview"
        assert saved["application_notes"] == "Prepare system design examples."
        assert len(saved["application_events"]) == 1
        assert saved["original_text"] == "Updated advertisement"
    finally:
        store.close()


def test_case_insensitive_company_identity_is_stable(tmp_path: Path) -> None:
    store = Store(tmp_path / "career.duckdb")
    try:
        first = store.create_job(_job(), "First ad", "run-1")
        second_job = _job()
        second_job.title = "Data Engineer"
        second_job.company = "ACME"
        second = store.create_job(second_job, "Second ad", "run-2")
        assert store.get_job(first)["company"] == "Acme"
        assert store.get_job(second)["company"] == "ACME"
        assert store.get_job(first)["company"] is not None
    finally:
        store.close()


def test_exported_backup_can_restore_database(tmp_path: Path) -> None:
    store = Store(tmp_path / "career.duckdb")
    store.create_job(_job(), "Original advertisement", "run-1")
    backup = tmp_path / "backup"
    store.backup(backup)
    store.restore(backup)
    assert len(store.list_jobs()) == 1
    store.close()
