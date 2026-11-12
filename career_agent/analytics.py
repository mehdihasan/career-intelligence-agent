from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from career_agent.llm import LLMProvider
from career_agent.storage import Store

CLOUD_SKILLS = {
    "aws", "azure", "gcp", "google cloud", "microsoft azure",
    "amazon web services", "google cloud platform",
}


def _top(values: dict[str, Any], count: int) -> dict[str, Any]:
    return dict(list(values.items())[:count])


class ReportGenerator:
    def __init__(self, store: Store, llm: LLMProvider):
        self.store = store
        self.llm = llm

    def generate(
        self,
        run_id: str,
        new_job_ids: list[str],
        profile_version: int,
    ) -> list[str]:
        rows = self.store.analytics_snapshot()
        all_report = self._build_report(rows, "all_jobs")
        new_set = set(new_job_ids)
        new_report = self._build_report(
            [row for row in rows if row["job_id"] in new_set], "new_jobs"
        )
        limitations = [
            "Statistics describe collected advertisements only, not the complete labor market.",
            "Scores and findings depend on available source text, model output, and this profile version.",
            "Unknown or unverified information is not evidence that a requirement or company feature is absent.",
        ]
        reports: list[dict[str, Any]] = []
        for report_type, payload in (("new_jobs", new_report), ("all_jobs", all_report)):
            payload.update({
                "report_type": report_type,
                "sample_size": payload["job_count"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "profile_version": profile_version,
                "limitations": limitations,
            })
            interpretation = self.llm.generate(
                self._interpretation_prompt(report_type, payload),
                max_new_tokens=1200,
            )
            payload["interpretation"] = interpretation
            reports.append({
                "run_id": run_id,
                "report_type": report_type,
                "sample_size": payload["job_count"],
                "profile_version": profile_version,
                "limitations": limitations,
                "report_json": payload,
            })
        return self.store.save_reports(reports)

    @staticmethod
    def _build_report(rows: list[dict[str, Any]], report_type: str) -> dict[str, Any]:
        jobs = [row for row in rows if row.get("extraction_json")]
        skill_frequency: Counter[str] = Counter()
        required: Counter[str] = Counter()
        preferred: Counter[str] = Counter()
        language_frequency: Counter[str] = Counter()
        source_language_frequency: Counter[str] = Counter()
        role_frequency: Counter[str] = Counter()
        geography: Counter[str] = Counter()
        cloud: Counter[str] = Counter()
        matches: Counter[str] = Counter()
        gaps: Counter[str] = Counter()
        unverified: Counter[str] = Counter()
        learning_actions: Counter[str] = Counter()
        reasons_to_apply: Counter[str] = Counter()
        reasons_for_caution: Counter[str] = Counter()
        score_values: list[float] = []
        opportunity_values: list[float] = []
        application_pipeline: Counter[str] = Counter()
        period_skills: dict[str, Counter[str]] = defaultdict(Counter)
        top_opportunities: list[dict[str, Any]] = []

        for row in jobs:
            extracted = row["extraction_json"]
            title = row.get("title") or "Unknown role"
            role_frequency[title] += 1
            place = ", ".join(filter(None, (row.get("city"), row.get("country")))) or "Unknown"
            geography[place] += 1
            application_pipeline[row.get("application_status") or "not_applied"] += 1
            if extracted.get("source_language"):
                source_language_frequency[extracted["source_language"]] += 1
            imported_at = row.get("imported_at")
            period = imported_at.strftime("%Y-%m") if imported_at else "Unknown"
            for skill in extracted.get("skills", []):
                name = skill.get("canonical_id") or skill.get("name", "unknown")
                skill_frequency[name] += 1
                period_skills[period][name] += 1
                if skill.get("requirement") == "preferred":
                    preferred[name] += 1
                else:
                    required[name] += 1
                if name.casefold() in CLOUD_SKILLS:
                    cloud[name] += 1
            for language in extracted.get("languages", []):
                language_frequency[language] += 1
            evaluation = row.get("evaluation_json")
            if not evaluation:
                continue
            if evaluation.get("job_fit_score") is not None:
                score_values.append(float(evaluation["job_fit_score"]))
            opportunity_score = evaluation.get("overall_opportunity_score")
            if opportunity_score is not None:
                opportunity_values.append(float(opportunity_score))
                top_opportunities.append({
                    "job_id": row["job_id"],
                    "company": row.get("company"),
                    "title": title,
                    "score": float(opportunity_score),
                })
            for skill in evaluation.get("matched_skills", []):
                matches[skill] += 1
            for skill in evaluation.get("likely_missing_skills", []):
                gaps[skill] += 1
            for skill in evaluation.get("unverified_skills", []):
                unverified[skill] += 1
            for action in evaluation.get("learning_actions", []):
                learning_actions[action] += 1
            for reason in evaluation.get("reasons_to_apply", []):
                reasons_to_apply[reason] += 1
            for reason in evaluation.get("reasons_for_caution", []):
                reasons_for_caution[reason] += 1

        sample_size = len(jobs)
        return {
            "job_count": len(jobs),
            "role_distribution": _counter_dict(role_frequency),
            "role_distribution_percent": _percent_dict(role_frequency, sample_size),
            "skill_frequency": _counter_dict(skill_frequency),
            "skill_frequency_percent": _percent_dict(skill_frequency, sample_size),
            "mandatory_skills": _counter_dict(required),
            "mandatory_skills_percent": _percent_dict(required, sample_size),
            "preferred_skills": _counter_dict(preferred),
            "preferred_skills_percent": _percent_dict(preferred, sample_size),
            "cloud_platform_distribution": _counter_dict(cloud),
            "cloud_platform_distribution_percent": _percent_dict(cloud, sample_size),
            "language_requirements": _counter_dict(language_frequency),
            "language_requirements_percent": _percent_dict(language_frequency, sample_size),
            "advertisement_source_languages": _counter_dict(source_language_frequency),
            "geographic_distribution": _counter_dict(geography),
            "geographic_distribution_percent": _percent_dict(geography, sample_size),
            "matched_skills": _counter_dict(matches),
            "recurring_skill_gaps": _counter_dict(gaps),
            "unverified_skills": _counter_dict(unverified),
            "learning_recommendations": _counter_dict(learning_actions),
            "reasons_to_apply": _counter_dict(reasons_to_apply),
            "reasons_for_caution": _counter_dict(reasons_for_caution),
            "job_fit_score_distribution": _score_distribution(score_values),
            "opportunity_score_distribution": _score_distribution(opportunity_values),
            "application_pipeline": _counter_dict(application_pipeline),
            "application_pipeline_percent": _percent_dict(application_pipeline, sample_size),
            "top_opportunities": sorted(
                top_opportunities, key=lambda value: value["score"], reverse=True
            )[:10],
            "skill_frequency_by_import_period": {
                period: _counter_dict(counter)
                for period, counter in sorted(period_skills.items())
            },
            "report_scope": report_type,
        }

    @staticmethod
    def _interpretation_prompt(report_type: str, stats: dict[str, Any]) -> str:
        compact = {
            "sample_size": stats["sample_size"],
            "role_distribution": _top(stats.get("role_distribution", {}), 20),
            "skill_frequency": _top(stats.get("skill_frequency", {}), 30),
            "mandatory_skills": _top(stats.get("mandatory_skills", {}), 20),
            "preferred_skills": _top(stats.get("preferred_skills", {}), 20),
            "cloud_platform_distribution": _top(
                stats.get("cloud_platform_distribution", {}), 10
            ),
            "language_requirements": _top(stats.get("language_requirements", {}), 15),
            "geographic_distribution": _top(stats.get("geographic_distribution", {}), 15),
            "matched_skills": _top(stats.get("matched_skills", {}), 20),
            "recurring_skill_gaps": _top(stats.get("recurring_skill_gaps", {}), 20),
            "learning_recommendations": _top(
                stats.get("learning_recommendations", {}), 15
            ),
            "top_opportunities": stats.get("top_opportunities", [])[:10],
            "application_pipeline": stats.get("application_pipeline", {}),
            "job_fit_score_distribution": stats.get("job_fit_score_distribution"),
            "opportunity_score_distribution": stats.get(
                "opportunity_score_distribution"
            ),
            "skill_frequency_by_import_period": {
                period: _top(counts, 10)
                for period, counts in list(
                    stats.get("skill_frequency_by_import_period", {}).items()
                )[-12:]
            },
        }
        return f"""Interpret calculated statistics for a career decision-support report.
The scope is {report_type}. The supplied numbers were computed deterministically; do not alter,
recalculate, or invent them. Clearly distinguish advertisement sample patterns from labor-market
claims. Highlight notable role and skill demand, evidence-based matches and gaps, top opportunities,
and practical learning suggestions grounded in these statistics. State uncertainty where samples
are small. If the sample is empty, say no new jobs were collected and do not invent trends.

CALCULATED DATA:
{compact}"""


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(counter.most_common())


def _percent_dict(counter: Counter[str], denominator: int) -> dict[str, float]:
    if not denominator:
        return {}
    return {
        key: round(count * 100 / denominator, 2)
        for key, count in counter.most_common()
    }


def _score_distribution(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "bands": {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0},
        }
    bands = {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    for score in scores:
        band = min(int(score // 20), 4)
        lower = band * 20
        upper = min(lower + 19, 100)
        bands[f"{lower}-{upper}"] += 1
    return {
        "count": len(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "mean": round(sum(scores) / len(scores), 2),
        "bands": bands,
    }
