# Career Intelligence Agent

## Software Requirements Specification — v1.0

**Status:** MVP specification
**Deployment:** Local-first on macOS or Linux; Apple Silicon is recommended
**Memory:** Recommended 24–32 GB unified RAM on Apple Silicon; other devices depend on model, dtype, and context size
**Expected workload:** 10–30 job descriptions per run
**Operating model:** Explicit on-demand execution
**Internet research:** Free services only
**Interfaces:** CLI and Streamlit

---

## 1. Product objective

Build a local-first career intelligence application that analyzes job advertisements against a user's CV, skills, career objectives, and professional preferences.

The system must help the user:

1. Identify attractive career opportunities.
2. Understand how well their background matches job requirements.
3. Discover recurring skill gaps.
4. Identify emerging technologies and hiring patterns.
5. Compare potential employers and engineering environments.
6. Track job applications and interview outcomes.
7. Make evidence-based decisions about future learning and career moves.

The system is a career decision-support tool, not an automated job application platform.

## 2. Core architecture

The MVP shall consist of:

- One local application backend.
- One shared local LLM runtime.
- Three sequential LLM-powered agents.
- Deterministic ingestion, validation, scoring aggregation, and persistence services.
- A DuckDB database.
- A Streamlit dashboard.
- A Typer CLI.
- Optional MCP-based research tools.

The application shall run without Docker, Kubernetes, external LLM APIs, or paid cloud services.

### 2.1 Execution model

Both interfaces shall invoke the same backend.

Recommended components:

- FastAPI for localhost application endpoints.
- Typer for CLI commands.
- Streamlit for the dashboard.
- A single in-process execution queue for analysis runs.

Only one analysis run shall execute at a time.

The backend shall own the LLM instance and DuckDB write access.

CLI and Streamlit shall not open independent write connections to the database.

The application shall listen on localhost by default and shall not be exposed to the public network.

## 3. Input requirements

### 3.1 Career profile

The system shall accept a YAML configuration describing:

- Target roles.
- Target technologies.
- Existing skills.
- Desired skills.
- Career interests.
- Location preferences.
- Remote/hybrid preferences.
- Working languages.
- Salary expectations, if provided.
- Career-growth priorities.
- Hard constraints.

The career profile shall be editable from the dashboard.

### 3.2 CV

The system shall accept TXT, Markdown, PDF, and DOCX CV files.

The CV shall be converted into a structured professional profile.

Extracted experience shall include employment history, technologies, projects, achievements, education, and certifications.

The user must be able to review and correct extracted information.

A skill absent from the CV shall be classified as **unverified**, not automatically missing.

### 3.3 Job directory

The system shall accept TXT, Markdown, and HTML files for the MVP.

Each file normally represents one advertisement.

Advertisements may originate from LinkedIn, Arbetsförmedlingen, BDJobs, Canadian portals, or employer websites.

The parser shall tolerate copied navigation text, inconsistent headings, missing fields, and different languages.

Original job descriptions shall be preserved.

All normalized analytical fields shall be stored in English.

## 4. Local LLM

### 4.1 Default model

Use:

`Qwen/Qwen3-4B-Instruct-2507`

Initial runtime:

- Hugging Face Transformers.
- PyTorch.
- Accelerate.
- Apple Silicon MPS by default; CPU or compatible CUDA devices can be selected in settings.
- BF16 by default; configure a dtype supported by the selected device.
- Sequential requests.
- One shared model instance.

The model shall be loaded lazily when first required and reused during a run.

### 4.2 Model abstraction

Implement an `LLMProvider` interface with operations for:

- Text generation.
- Structured extraction.
- Translation.
- Summarization.
- Evaluation.

The application shall not depend directly on one model's output format.

Future backends may include MLX-LM, llama.cpp, or an explicitly enabled hosted model.

### 4.3 Inference safeguards

- Validate structured output with Pydantic.
- Use deterministic decoding where supported.
- Retry malformed outputs up to two times.
- Bound context and output length.
- Split oversized documents into manageable sections.
- Record inference failures.
- Never invent absent job information.
- Reuse cached results when inputs and configuration are unchanged.

## 5. Agent 1 — Job Extraction

### Responsibilities

Process each eligible job file and produce normalized structured data.

Extract:

- Company.
- Job title.
- Source URL and reference ID.
- Country and city.
- Work mode.
- Employment type.
- Seniority.
- Salary and currency.
- Required experience.
- Responsibilities.
- Mandatory technical skills.
- Preferred technical skills.
- Education and certifications.
- Language requirements.
- Benefits.
- Application deadline.
- English summary.

### Skill normalization

Normalize equivalent technology names into canonical identifiers.

Examples:

- PostgreSQL / Postgres.
- Kubernetes / K8s.
- Amazon Web Services / AWS.

Preserve the distinction between mandatory and optional skills.

### Output requirements

The agent shall return a Pydantic-validated object.

Important extracted claims shall reference supporting source passages.

Uncertain or absent values shall be represented explicitly.

## 6. Deduplication service

Deduplication shall use deterministic methods before invoking an LLM.

### Stage A: Exact matching

SHA-256 hash of original file contents.

### Stage B: Normalized content matching

Normalize whitespace, copied navigation text, and formatting before calculating a fingerprint.

### Stage C: Identity matching

Compare normalized company, title, location, source reference, and text similarity.

### Stage D: Ambiguous matches

Flag uncertain duplicates for manual review.

The system must distinguish:

- Same job copied twice.
- Same job copied from different websites.
- Updated version of an existing advertisement.
- New vacancy with a similar title.
- Reposted vacancy.

Repeated runs shall not create duplicate canonical jobs.

A changed source file shall not overwrite application notes or interview history.

## 7. Agent 2 — Company Research

### Responsibilities

Research publicly available information about the employer and, where possible, the relevant engineering team.

Research categories:

- Company products and business.
- Industry.
- Company locations.
- Approximate company size.
- Public engineering technology stack.
- Engineering blogs.
- Public GitHub projects.
- Team responsibilities.
- Engineering practices.
- Recent relevant developments.
- Potential learning opportunities.
- Available company stability indicators.

### Research implementation

The MVP shall implement a `ResearchTool` abstraction.

It shall support:

- Search.
- Fetch webpage.
- Extract readable text.
- Return source metadata.

A free search/fetch adapter shall be implemented first.

An MCP adapter may be added without changing agent logic.

Free search availability and rate limits shall be treated as external dependencies. The application shall not rely on a guaranteed unlimited free search API.

### Research restrictions

- No login automation.
- No CAPTCHA bypass.
- No unauthorized scraping.
- No private employee information gathering.
- No paid API requirement.
- Respect website access restrictions.
- Bound search requests and timeouts.

Default research budget:

- Maximum 5 search queries per new company.
- Maximum 10 retrieved pages.
- Maximum 60 seconds per company, configurable.

### Evidence requirements

Every company research finding shall include source URLs and retrieval timestamps.

Unsupported claims shall be marked unknown.

Company-level research shall be cached and reused across vacancies.

## 8. Agent 3 — Opportunity Evaluation

The evaluation agent shall compare each job with the active career profile.

### Evaluation dimensions

1. Technical fit.
2. Experience and seniority fit.
3. Career direction alignment.
4. Learning potential.
5. Engineering environment.
6. Practical constraints.
7. Overall opportunity attractiveness.

The agent shall produce evidence-based assessments for each dimension.

A deterministic scoring service shall calculate weighted scores between 0 and 100.

Scoring weights shall be configurable.

### Fit versus opportunity

The system must distinguish:

**Job fit:** How qualified is the user for this position?

**Opportunity quality:** How attractive is the position for the user's future growth?

A job with a moderate technical match but exceptional learning potential may receive a high opportunity score.

### Missing information

Unknown company culture, mentorship, compensation, or technology details shall not be treated as proven negatives.

The evaluation shall indicate evidence coverage and uncertainty.

### Required outputs

- Technical fit score.
- Career alignment score.
- Learning potential score.
- Overall opportunity score.
- Matched skills.
- Partially matched skills.
- Unverified skills.
- Likely missing skills.
- Potential blockers.
- Reasons to apply.
- Reasons for caution.
- Suggested interview questions.
- Recommended learning actions.

## 9. Persistence

Use DuckDB.

### Required tables

- `career_profiles`
- `jobs`
- `job_sources`
- `job_skills`
- `companies`
- `company_research`
- `research_sources`
- `job_evaluations`
- `applications`
- `application_events`
- `processing_runs`
- `processing_errors`
- `analysis_reports`

### Application tracking

Applications shall support:

- Not applied.
- Interested.
- Preparing.
- Applied.
- Recruiter screening.
- Technical interview.
- Hiring manager interview.
- Final interview.
- Offer.
- Rejected.
- Withdrawn.
- Accepted.

Each application shall have an ordered event history.

The user shall be able to record interview dates, feedback, notes, and outcomes.

### Data integrity

Use stable identifiers, unique constraints, and transactional writes.

Preserve source content and historical evaluations.

Manual application information shall never be overwritten by automated analysis.

## 10. On-demand processing

The CLI shall support:

```bash
career-agent run
career-agent status
career-agent jobs list
career-agent jobs show <job-id>
career-agent reports list
career-agent serve
```

The dashboard shall provide a **Run Analysis** action.

Each run shall:

1. Register a processing run.
2. Discover files.
3. Detect unchanged inputs.
4. Extract new or changed advertisements.
5. Resolve duplicates.
6. Research companies where necessary.
7. Evaluate jobs.
8. Persist validated results.
9. Calculate aggregate statistics.
10. Generate two reports.
11. Record completion and errors.

Each job shall have an independent processing state.

Failed jobs shall be retryable.

A run with zero new jobs shall still generate the required reports.

## 11. Report generation

Each run shall generate exactly two primary reports.

### Report A: New Jobs

Analyze only newly accepted canonical jobs in the current run.

Include:

- Total new jobs.
- Job categories.
- Most requested technologies.
- Mandatory versus optional skills.
- Cloud-platform distribution.
- Language requirements.
- Geographic distribution.
- User skill matches.
- Recurring skill gaps.
- Top opportunities.
- Learning recommendations.

### Report B: All Jobs

Analyze all canonical jobs in the database.

Include:

- Total job count.
- Role distribution.
- Skill frequency.
- Changes across import periods.
- Geographic distribution.
- Cloud and data-platform trends.
- Technical fit distribution.
- Opportunity-score distribution.
- Recurring skill gaps.
- Application pipeline.
- Career-development recommendations.

### Analytical correctness

Counts, percentages, distributions, and weighted scores shall be computed using SQL or Python.

The LLM shall interpret the calculated results.

Reports shall distinguish collected-advertisement trends from claims about the entire labor market.

Every report shall record its sample size, date, profile version, and analysis limitations.

Reports shall be immutable historical snapshots.

## 12. Dashboard

Use Streamlit.

### Page 1: Job Explorer

Display a sortable, searchable job list.

Fields:

- Company.
- Role.
- Location.
- Technical fit.
- Opportunity score.
- Learning potential.
- Application status.
- Import date.

Support filtering by role, skill, location, language, score, and status.

### Page 2: Job Details

Display:

- Original advertisement.
- Normalized English description.
- Extracted information.
- Company research.
- Research evidence.
- Fit and opportunity scores.
- Skill-gap analysis.
- Recommendations.
- Application timeline.

Allow application-status and note editing.

### Page 3: Reports

Display historical reports and their details.

Support charts and tables for aggregate statistics.

### Page 4: Market Insights

Display:

- Most frequent skills.
- Skill frequency by role.
- Geographic patterns.
- Language requirements.
- Cloud technology distributions.
- Skill gaps.
- Trends across processing runs.

### Page 5: Application Tracker

Display applications and interview events.

Allow manual updates and chronological tracking.

### Page 6: Settings and Runs

Display:

- Career profile.
- CV settings.
- Input directory.
- Model settings.
- Research limits.
- Scoring weights.
- Run history.
- Processing errors.

## 13. Privacy and security

The application shall:

- Keep CV data local.
- Run inference locally.
- Avoid sending CV contents to research services.
- Send only minimal company/job queries to external tools.
- Treat job advertisements and webpages as untrusted data.
- Never execute instructions embedded in retrieved content.
- Restrict network research tools to permitted operations.
- Store secrets outside source code.
- Bind local services to loopback interfaces.
- Support database backup and restoration.

## 14. Performance

The system shall target batches of 10–30 advertisements.

Processing shall be sequential initially.

No fixed end-to-end runtime guarantee is required.

The application shall measure:

- Extraction latency.
- Research latency.
- Evaluation latency.
- Total run duration.
- Memory consumption.
- Model generation speed.
- Processing failures.

Results shall be used to decide whether quantization, caching improvements, or alternative runtimes are justified.

## 15. Testing and quality

Create a manually verified benchmark dataset containing at least 20 representative advertisements.

Include English, Swedish, Bengali, copied HTML, and inconsistent formatting.

Test:

- Extraction correctness.
- Required/preferred skill classification.
- Language detection.
- Translation fidelity.
- Duplicate detection.
- Missing-field handling.
- Company research attribution.
- Scoring consistency.
- Invalid model outputs.
- Research failures.
- Application history preservation.
- Repeat-run idempotency.

### Initial quality targets

- At least 95% accuracy for core fields on the labeled test set.
- At least 90% accuracy for required/preferred skill classification.
- No duplicate canonical records for exact duplicate inputs.
- No unsupported company claims presented as verified facts.
- No loss of manually entered application data.

These are acceptance targets to test, not assumed model capabilities.

## 16. Development phases

### Phase 1 — Local foundation

Implement model loading, configuration, CV parsing, job ingestion, extraction, validation, and DuckDB storage.

### Phase 2 — Intelligence pipeline

Implement deduplication, company research, scoring, evidence tracking, and failure recovery.

### Phase 3 — Reports

Implement aggregate SQL analytics and both historical report types.

### Phase 4 — Interfaces

Implement CLI commands, Streamlit dashboard, job details, reports, and application tracking.

### Phase 5 — Reliability

Benchmark extraction quality, improve prompts, add caching, test repeat runs, and optimize memory and performance.

## 17. Definition of done

The MVP is complete when:

1. A user can configure career goals and import a CV.
2. A batch of 10–30 heterogeneous job descriptions can be analyzed locally.
3. The application extracts normalized English job information.
4. Public company information is researched with traceable sources.
5. Jobs receive explainable fit, gap, and growth evaluations.
6. Duplicate advertisements do not create duplicate jobs.
7. All results persist in DuckDB.
8. Every run generates two historical reports.
9. Both CLI and Streamlit can initiate processing.
10. The dashboard supports job browsing, reports, and application tracking.
11. Repeated runs preserve application data.
12. No paid LLM or research API is required.

**Design principle:** Build a dependable, evidence-based career decision system. Avoid complexity that does not directly improve accuracy, reliability, or usability.
