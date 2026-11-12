from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WorkMode(StrEnum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"
    UNKNOWN = "unknown"


class SkillRequirement(StrictModel):
    name: str
    canonical_id: str | None = None
    requirement: Literal["mandatory", "preferred"]
    evidence: str | None = None


class EvidenceClaim(StrictModel):
    claim: str
    passage: str
    confidence: float = Field(ge=0, le=1)


class JobExtraction(StrictModel):
    company: str | None = None
    title: str | None = None
    source_language: str | None = None
    source_url: str | None = None
    reference_id: str | None = None
    country: str | None = None
    city: str | None = None
    work_mode: WorkMode = WorkMode.UNKNOWN
    employment_type: str | None = None
    seniority: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    required_experience_years: float | None = None
    responsibilities: list[str] = Field(default_factory=list)
    skills: list[SkillRequirement] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    application_deadline: date | None = None
    english_summary: str = ""
    normalized_description: str = ""
    claims: list[EvidenceClaim] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)


class Experience(StrictModel):
    employer: str | None = None
    title: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class ProfessionalProfile(StrictModel):
    summary: str = ""
    experience: list[Experience] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    unverified_skills: list[str] = Field(default_factory=list)


class CareerProfile(StrictModel):
    target_roles: list[str] = Field(default_factory=list)
    target_technologies: list[str] = Field(default_factory=list)
    existing_skills: list[str] = Field(default_factory=list)
    known_skill_gaps: list[str] = Field(default_factory=list)
    desired_skills: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    work_modes: list[str] = Field(default_factory=list)
    working_languages: list[str] = Field(default_factory=list)
    salary_min: float | None = None
    salary_currency: str | None = None
    growth_priorities: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    cv_profile: ProfessionalProfile = Field(default_factory=ProfessionalProfile)
    cv_filename: str | None = None
    updated_at: datetime | None = None


class ScoringWeights(StrictModel):
    technical_fit: float = Field(default=0.25, ge=0)
    experience_fit: float = Field(default=0.15, ge=0)
    career_alignment: float = Field(default=0.2, ge=0)
    learning_potential: float = Field(default=0.15, ge=0)
    engineering_environment: float = Field(default=0.1, ge=0)
    practical_constraints: float = Field(default=0.15, ge=0)


class ResearchSource(StrictModel):
    url: HttpUrl
    title: str
    retrieved_at: datetime
    category: str = "general"


class ResearchFinding(StrictModel):
    category: str
    finding: str
    evidence: str
    source_url: str
    confidence: float = Field(ge=0, le=1)


class CompanyResearch(StrictModel):
    company: str
    industry: str | None = None
    approximate_size: str | None = None
    locations: list[str] = Field(default_factory=list)
    findings: list[ResearchFinding] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )


class EvaluationDimension(StrictModel):
    score: float = Field(ge=0, le=100)
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class JobEvaluation(StrictModel):
    technical_fit: EvaluationDimension
    experience_fit: EvaluationDimension
    career_alignment: EvaluationDimension
    learning_potential: EvaluationDimension
    engineering_environment: EvaluationDimension
    practical_constraints: EvaluationDimension
    matched_skills: list[str] = Field(default_factory=list)
    partially_matched_skills: list[str] = Field(default_factory=list)
    unverified_skills: list[str] = Field(default_factory=list)
    likely_missing_skills: list[str] = Field(default_factory=list)
    potential_blockers: list[str] = Field(default_factory=list)
    reasons_to_apply: list[str] = Field(default_factory=list)
    reasons_for_caution: list[str] = Field(default_factory=list)
    interview_questions: list[str] = Field(default_factory=list)
    learning_actions: list[str] = Field(default_factory=list)
    evidence_coverage: float = Field(default=0, ge=0, le=1)
    uncertainty: list[str] = Field(default_factory=list)
    job_fit_score: float = Field(default=0, ge=0, le=100)
    overall_opportunity_score: float = Field(default=0, ge=0, le=100)


class ApplicationStatus(StrEnum):
    NOT_APPLIED = "not_applied"
    INTERESTED = "interested"
    PREPARING = "preparing"
    APPLIED = "applied"
    RECRUITER_SCREENING = "recruiter_screening"
    TECHNICAL_INTERVIEW = "technical_interview"
    HIRING_MANAGER_INTERVIEW = "hiring_manager_interview"
    FINAL_INTERVIEW = "final_interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    ACCEPTED = "accepted"


class ApplicationUpdate(StrictModel):
    status: ApplicationStatus | None = None
    notes: str | None = None
    event_type: str | None = None
    event_date: datetime | None = None
    feedback: str | None = None


class RunRequest(StrictModel):
    input_directory: str | None = None
    force: bool = False


class RunSummary(StrictModel):
    run_id: str
    status: str
    discovered: int = 0
    new_jobs: int = 0
    updated_jobs: int = 0
    unchanged_jobs: int = 0
    errors: int = 0
    started_at: datetime
    completed_at: datetime | None = None
    details: dict[str, Any] = Field(default_factory=dict)
