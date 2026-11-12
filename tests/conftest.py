from __future__ import annotations

from typing import Any

import pytest

from career_agent.schemas import (
    EvaluationDimension,
    JobEvaluation,
    JobExtraction,
    SkillRequirement,
)


class FakeLLM:
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return "Summary grounded in the calculated report data."

    def structured(self, prompt: str, schema: type[Any], max_new_tokens: int | None = None) -> Any:
        if schema is JobExtraction:
            return JobExtraction(
                company="Acme",
                title="Backend Engineer",
                english_summary="Build backend services.",
                normalized_description="Build backend services with PostgreSQL.",
                skills=[
                    SkillRequirement(
                        name="PostgreSQL",
                        requirement="mandatory",
                        evidence="PostgreSQL required",
                    )
                ],
            )
        if schema is JobEvaluation:
            dimension = EvaluationDimension(score=75, rationale="Supported by the supplied evidence.")
            return JobEvaluation(
                technical_fit=dimension,
                experience_fit=dimension,
                career_alignment=dimension,
                learning_potential=dimension,
                engineering_environment=dimension,
                practical_constraints=dimension,
            )
        raise AssertionError(f"Unexpected structured schema: {schema.__name__}")


class EmptyResearchTool:
    def search(self, query: str, limit: int = 5, timeout: float = 10) -> list[Any]:
        return []

    def fetch(self, url: str, timeout: float = 10) -> None:
        return None


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def empty_research() -> EmptyResearchTool:
    return EmptyResearchTool()
