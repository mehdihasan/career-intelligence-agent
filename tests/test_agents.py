from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from career_agent.agents import (
    CompanyResearchAgent,
    JobExtractionAgent,
    JobEvaluationAgent,
)
from career_agent.config import ModelSettings, ResearchSettings
from career_agent.llm import TransformersProvider
from career_agent.research import ResearchPage, SearchResult
from career_agent.schemas import (
    CareerProfile,
    CompanyResearch,
    EvaluationDimension,
    JobEvaluation,
    JobExtraction,
    ProfessionalProfile,
    ResearchFinding,
    ScoringWeights,
)
from career_agent.storage import Store


class CitedResearchTool:
    def __init__(self) -> None:
        self.searches = 0

    def search(self, query: str, limit: int, timeout: float) -> list[SearchResult]:
        self.searches += 1
        return [SearchResult("https://example.com/engineering", "Engineering")]

    def fetch(self, url: str, timeout: float) -> ResearchPage:
        return ResearchPage(
            url=url,
            title="Engineering",
            text="Acme builds financial software for local businesses.",
            retrieved_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )


class ResearchLLM:
    def structured(self, prompt: str, schema: type[Any], max_new_tokens: int | None = None) -> Any:
        return schema(
            industry="Financial technology",
            findings=[
                ResearchFinding(
                    category="products",
                    finding="Builds financial software.",
                    evidence="Acme builds financial software",
                    source_url="https://example.com/engineering",
                    confidence=0.9,
                ),
                ResearchFinding(
                    category="company_size",
                    finding="Employs 10,000 people.",
                    evidence="The page has no headcount evidence.",
                    source_url="https://example.com/engineering",
                    confidence=1.0,
                ),
            ],
        )


class EvaluationLLM:
    def structured(self, prompt: str, schema: type[Any], max_new_tokens: int | None = None) -> Any:
        scores = {
            "technical_fit": 80,
            "experience_fit": 40,
            "career_alignment": 60,
            "learning_potential": 100,
            "engineering_environment": 20,
            "practical_constraints": 70,
        }
        dimensions = {
            name: EvaluationDimension(score=score, rationale="Evidence-based assessment.")
            for name, score in scores.items()
        }
        return JobEvaluation(**dimensions)


class RetryProvider(TransformersProvider):
    def __init__(self, outputs: list[str]):
        super().__init__(ModelSettings())
        self.outputs = outputs
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def test_invalid_structured_output_is_retried_twice() -> None:
    provider = RetryProvider([
        "not json",
        '{"title": 4}',
        '{"company":"Acme","title":"Backend Engineer"}',
    ])
    result = provider.structured("extract", JobExtraction)
    assert result.company == "Acme"
    assert len(provider.prompts) == 3
    assert "Previous response" in provider.prompts[1]


def test_structured_generation_supplies_schema_and_retries_invalid_cv_fields() -> None:
    provider = RetryProvider([
        '{"core_skills":["Python"],"experiences":[],"references":null}',
        '{"skills":["Python"],"experience":[{"employer":"Acme"}]}',
    ])
    result = provider.structured("Extract CV", ProfessionalProfile)
    assert result.skills == ["Python"]
    assert result.experience[0].employer == "Acme"
    expected_schema = json.dumps(
        ProfessionalProfile.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    assert len(provider.prompts) == 2
    assert all(expected_schema in prompt for prompt in provider.prompts)
    assert "core_skills" in provider.prompts[1]


def test_transformers_generation_uses_tensor_chat_template_output() -> None:
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    tokenizer_model = tokenizers.models.WordLevel(
        {"[UNK]": 0, "prompt": 1, "answer": 2}, unk_token="[UNK]"
    )
    tokenizer_object = tokenizers.Tokenizer(tokenizer_model)
    tokenizer_object.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        unk_token="[UNK]",
        eos_token="[UNK]",
        chat_template="{{ messages[-1]['content'] }}",
    )

    class Model:
        device = "cpu"

        def generate(self, input_ids: Any, **kwargs: Any) -> Any:
            return torch.cat((input_ids, torch.tensor([[2]])), dim=-1)

    provider = TransformersProvider(ModelSettings(device="cpu", dtype="float32"))
    provider._tokenizer = tokenizer
    provider._model = Model()
    assert provider.generate("prompt") == "answer"
    assert provider.metrics()["generated_tokens"] == 1


def test_extracted_source_url_must_exist_in_source_document() -> None:
    class HallucinatedURLLLM:
        def structured(self, prompt: str, schema: type[Any], max_new_tokens: int | None = None) -> Any:
            return JobExtraction(
                title="Backend Engineer",
                source_url="https://unsupported.example/jobs/1",
            )

    extraction = JobExtractionAgent(HallucinatedURLLLM(), ModelSettings()).extract(
        "Backend Engineer opening"
    )
    assert extraction.source_url is None
    assert "source_url" in extraction.uncertain_fields


def test_company_research_requires_retrieved_evidence_and_caches(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "career.duckdb")
    try:
        tool = CitedResearchTool()
        agent = CompanyResearchAgent(
            ResearchLLM(),
            tool,
            ResearchSettings(max_queries_per_company=1, max_pages_per_company=1),
            ModelSettings(),
            store,
        )
        agent.begin_run()
        result = agent.research("Acme").research
        assert len(result.findings) == 1
        assert result.findings[0].evidence == "Acme builds financial software"
        assert len(result.sources) == 1
        assert "company_size" not in result.unknowns
        agent.research("Acme")
        assert tool.searches == 1
        changed_settings = ResearchSettings(
            max_queries_per_company=2, max_pages_per_company=1
        )
        updated_agent = CompanyResearchAgent(
            ResearchLLM(), tool, changed_settings, ModelSettings(), store
        )
        updated_agent.begin_run()
        updated_agent.research("Acme")
        assert tool.searches > 1
    finally:
        store.close()


def test_weighted_opportunity_and_job_fit_scores_are_deterministic() -> None:
    from career_agent.schemas import ScoringWeights

    agent = JobEvaluationAgent(
        EvaluationLLM(),
        ScoringWeights(
            technical_fit=1,
            experience_fit=1,
            career_alignment=0,
            learning_potential=0,
            engineering_environment=0,
            practical_constraints=0,
        ),
        max_input_chars=24000,
    )
    result = agent.evaluate(JobExtraction(), CareerProfile(), CompanyResearch(company="Acme"))
    assert result.overall_opportunity_score == 60
    assert result.job_fit_score == 66


def test_absent_skills_are_unverified_unless_the_profile_confirms_a_gap() -> None:
    class SkillEvaluationLLM(EvaluationLLM):
        def structured(self, prompt: str, schema: type[Any], max_new_tokens: int | None = None) -> Any:
            result = super().structured(prompt, schema, max_new_tokens)
            result.matched_skills = ["Postgres"]
            result.likely_missing_skills = ["K8s", "Python"]
            return result

    job = JobExtraction(
        skills=[
            {"name": "PostgreSQL", "requirement": "mandatory"},
            {"name": "Kubernetes", "requirement": "mandatory"},
            {"name": "Python", "requirement": "preferred"},
        ]
    )
    agent = JobEvaluationAgent(
        SkillEvaluationLLM(),
        ScoringWeights(),
        max_input_chars=24000,
    )
    profile = CareerProfile(existing_skills=["PostgreSQL"])
    result = agent.evaluate(job, profile, None)
    assert result.matched_skills == ["postgresql"]
    assert result.likely_missing_skills == []
    assert result.unverified_skills == ["kubernetes", "python"]

    profile.known_skill_gaps = ["K8s"]
    result = agent.evaluate(job, profile, None)
    assert result.likely_missing_skills == ["kubernetes"]
    assert result.unverified_skills == ["python"]
