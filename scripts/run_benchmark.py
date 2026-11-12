from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from career_agent.config import load_settings
from career_agent.ingestion import canonical_skill, clean_job_text, html_to_text
from career_agent.llm import TransformersProvider
from career_agent.agents import JobExtractionAgent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure local job-extraction accuracy against the labeled fixture."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "fixtures" / "benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the detailed JSON result.",
    )
    args = parser.parse_args()
    benchmark = json.loads(args.dataset.read_text(encoding="utf-8"))
    settings = load_settings()
    agent = JobExtractionAgent(TransformersProvider(settings.model), settings.model)
    correct_fields = 0
    field_count = 0
    skill_correct = 0
    skill_count = 0
    failures: list[dict[str, str]] = []
    cases: list[dict[str, Any]] = []

    for item in benchmark["ads"]:
        content = item["content"]
        text = html_to_text(content) if item["format"] == "html" else content
        try:
            extracted = agent.extract(clean_job_text(text))
        except Exception as error:
            failures.append({"id": item["id"], "error": str(error)})
            continue
        expected = item["expected"]
        field_results: dict[str, bool] = {}
        for field in ("company", "title", "country", "source_language"):
            if field not in expected:
                continue
            actual_value = getattr(extracted, field)
            matches = _normalize(actual_value) == _normalize(expected[field])
            field_results[field] = matches
            correct_fields += int(matches)
            field_count += 1
        expected_skills = {
            canonical_skill(name): requirement
            for requirement, names in (
                ("mandatory", expected["required_skills"]),
                ("preferred", expected["preferred_skills"]),
            )
            for name in names
        }
        actual_skills = {
            canonical_skill(skill.canonical_id or skill.name): skill.requirement
            for skill in extracted.skills
        }
        for name in expected_skills.keys() | actual_skills.keys():
            skill_correct += int(expected_skills.get(name) == actual_skills.get(name))
            skill_count += 1
        cases.append({
            "id": item["id"],
            "field_results": field_results,
            "expected_skills": expected_skills,
            "actual_skills": actual_skills,
            "translated_summary_present": bool(extracted.english_summary.strip()),
        })

    core_accuracy = correct_fields / field_count if field_count else 0
    skill_accuracy = skill_correct / skill_count if skill_count else 0
    report = {
        "dataset": str(args.dataset),
        "cases": len(benchmark["ads"]),
        "completed": len(cases),
        "core_field_accuracy": round(core_accuracy, 4),
        "required_preferred_skill_accuracy": round(skill_accuracy, 4),
        "targets": {"core_fields": 0.95, "required_preferred_skills": 0.90},
        "target_results": {
            "core_fields_met": core_accuracy >= 0.95,
            "required_preferred_skills_met": skill_accuracy >= 0.90,
        },
        "failures": failures,
        "case_results": cases,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all(report["target_results"].values()) else 1


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


if __name__ == "__main__":
    sys.exit(main())
