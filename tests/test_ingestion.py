import json
import hashlib
from pathlib import Path

from career_agent.ingestion import (
    canonical_skill,
    clean_job_text,
    normalized_hash,
    read_job_file,
    split_text,
)


def test_normalized_hash_ignores_formatting_and_common_navigation() -> None:
    first = "Backend engineer\n\nPostgreSQL required. Apply now"
    second = "BACKEND   ENGINEER PostgreSQL required"
    assert normalized_hash(first) == normalized_hash(second)


def test_skill_aliases_use_canonical_identifiers() -> None:
    assert canonical_skill("K8s") == "kubernetes"
    assert canonical_skill("Amazon Web Services") == "aws"
    assert canonical_skill("PostgreSQL") == "postgresql"


def test_split_text_bounds_chunks_and_preserves_input_content() -> None:
    text = "\n\n".join(f"Paragraph {index}: " + "x" * 45 for index in range(8))
    chunks = split_text(text, max_chars=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    for index in range(8):
        paragraph = f"Paragraph {index}: " + "x" * 45
        assert any(paragraph in chunk for chunk in chunks)


def test_clean_job_text_removes_duplicate_navigation_lines() -> None:
    assert clean_job_text("Apply now\nBackend role\nBackend role\nSign in") == "Backend role"


def test_exact_hash_uses_original_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.txt"
    original_bytes = b"Backend role \xff"
    path.write_bytes(original_bytes)
    document = read_job_file(path)
    assert document.exact_hash == hashlib.sha256(original_bytes).hexdigest()


def test_benchmark_fixture_has_twenty_labeled_multilingual_ads() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "benchmark.json"
    benchmark = json.loads(fixture_path.read_text(encoding="utf-8"))
    ads = benchmark["ads"]
    assert len(ads) >= 20
    assert {item["language"] for item in ads} >= {"en", "sv", "bn"}
    assert "html" in {item["format"] for item in ads}
    for item in ads:
        expected = item["expected"]
        assert expected["company"] and expected["title"]
        assert expected["source_language"] == item["language"]
        assert set(expected["required_skills"]).isdisjoint(expected["preferred_skills"])
