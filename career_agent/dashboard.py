from __future__ import annotations

import base64
import json
import os
from datetime import date, datetime, time
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("CAREER_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api(method: str, path: str, **kwargs: Any) -> Any:
    timeout = (30, None) if method == "POST" and path == "/profile/cv" else 30
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as error:
        st.error(f"Cannot reach the local career-agent API at {API_URL}: {error}")
        st.stop()
    if not response.ok:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        st.error(f"API request failed ({response.status_code}): {detail}")
        st.stop()
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Career Intelligence Agent", layout="wide")
    st.title("Career Intelligence Agent")
    page = st.sidebar.radio(
        "Workspace",
        [
            "Job Explorer",
            "Job Details",
            "Reports",
            "Market Insights",
            "Application Tracker",
            "Settings and Runs",
        ],
    )
    if page == "Job Explorer":
        show_job_explorer()
    elif page == "Job Details":
        show_job_details()
    elif page == "Reports":
        show_reports()
    elif page == "Market Insights":
        show_market_insights()
    elif page == "Application Tracker":
        show_application_tracker()
    else:
        show_settings_and_runs()


def show_job_explorer() -> None:
    st.header("Job Explorer")
    with st.expander("Filters", expanded=True):
        first, second, third = st.columns(3)
        search = first.text_input("Company, role, or location")
        role = second.text_input("Role contains")
        skill = third.text_input("Skill contains")
        fourth, fifth, sixth = st.columns(3)
        location = fourth.text_input("Location contains")
        language = fifth.text_input("Language contains")
        status_filter = sixth.selectbox(
            "Application status",
            ["Any", "not_applied", "interested", "preparing", "applied",
             "recruiter_screening", "technical_interview", "hiring_manager_interview",
             "final_interview", "offer", "rejected", "withdrawn", "accepted"],
        )
        minimum_score = st.slider("Minimum opportunity score", 0, 100, 0)
    params = {
        key: value
        for key, value in {
            "search": search,
            "role": role,
            "skill": skill,
            "location": location,
            "language": language,
            "status_filter": None if status_filter == "Any" else status_filter,
            "minimum_score": minimum_score if minimum_score else None,
            "limit": 1000,
        }.items()
        if value is not None and value != ""
    }
    jobs = api("GET", "/jobs", params=params)
    if not jobs:
        st.info("No matching jobs. Add TXT, Markdown, or HTML advertisements to the input directory and run analysis.")
        return
    display = [
        {
            "Company": item.get("company"),
            "Role": item.get("title"),
            "Location": ", ".join(filter(None, (item.get("city"), item.get("country")))),
            "Fit": item.get("job_fit_score"),
            "Opportunity": item.get("opportunity_score"),
            "Status": item.get("application_status"),
            "Imported": str(item.get("imported_at", ""))[:10],
            "Job ID": item["job_id"],
        }
        for item in jobs
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)
    selected_id = st.selectbox("Open job", [item["job_id"] for item in jobs])
    if st.button("View selected job"):
        st.session_state["selected_job_id"] = selected_id
        st.rerun()


def show_job_details() -> None:
    st.header("Job Details")
    jobs = api("GET", "/jobs", params={"limit": 1000})
    if not jobs:
        st.info("No jobs have been analyzed yet.")
        return
    ids = [item["job_id"] for item in jobs]
    default = st.session_state.get("selected_job_id", ids[0])
    selected = st.selectbox("Job", ids, index=ids.index(default) if default in ids else 0)
    job = api("GET", f"/jobs/{selected}")
    st.subheader(f"{job.get('title') or 'Untitled role'} - {job.get('company') or 'Unknown company'}")
    st.caption(", ".join(filter(None, (job.get("city"), job.get("country"), job.get("work_mode")))))
    extraction = job.get("extraction_json") or {}
    evaluation = job.get("evaluation")
    tabs = st.tabs(["Overview", "Evaluation", "Company Research", "Application"])
    with tabs[0]:
        st.markdown("#### English summary")
        st.write(job.get("english_summary") or "No summary was extracted.")
        st.markdown("#### Normalized description")
        st.write(job.get("normalized_description") or "No normalized description was extracted.")
        with st.expander("Original advertisement"):
            st.text(job.get("original_text", ""))
        with st.expander("Extracted fields and source history"):
            st.json(extraction)
            for source in job.get("sources", []):
                if source.get("duplicate_status") == "ambiguous":
                    st.warning(
                        "This source may duplicate job "
                        f"{source.get('potential_duplicate_job_id')}; compare both advertisements."
                    )
                st.caption(f"Source: {source.get('source_path')}")
                st.text(source.get("source_text", ""))
    with tabs[1]:
        if not evaluation:
            st.info("No evaluation is available for this job yet.")
        else:
            result = evaluation.get("evaluation_json", {})
            left, right = st.columns(2)
            left.metric("Job fit", result.get("job_fit_score", 0))
            right.metric("Opportunity", result.get("overall_opportunity_score", 0))
            st.json(result)
    with tabs[2]:
        research = job.get("research")
        if research:
            st.json(research)
            for source in research.get("sources", []):
                st.markdown(f"- [{source['title']}]({source['url']})")
        else:
            st.info("No verified public company research is available.")
    with tabs[3]:
        status_values = [
            "not_applied", "interested", "preparing", "applied", "recruiter_screening",
            "technical_interview", "hiring_manager_interview", "final_interview", "offer",
            "rejected", "withdrawn", "accepted",
        ]
        with st.form("application-update"):
            current_status = job.get("application_status") or "not_applied"
            status = st.selectbox(
                "Application status",
                status_values,
                index=status_values.index(current_status) if current_status in status_values else 0,
            )
            notes = st.text_area("Application notes", value=job.get("application_notes") or "")
            event_type = st.text_input("Interview or application event")
            event_date = st.date_input("Interview or application event date", value=date.today())
            feedback = st.text_area("Feedback or outcome")
            save = st.form_submit_button("Save application update")
        if save:
            api(
                "PATCH",
                f"/jobs/{selected}/application",
                json={
                    "status": status,
                    "notes": notes,
                    "event_type": event_type or None,
                    "event_date": datetime.combine(event_date, time.min).isoformat(),
                    "feedback": feedback or None,
                },
            )
            st.success("Application history saved.")
            st.rerun()
        st.markdown("#### Application timeline")
        st.json(job.get("application_events", []))


def show_reports() -> None:
    st.header("Historical Reports")
    reports = api("GET", "/reports")
    if not reports:
        st.info("Reports are created after an analysis run.")
        return
    st.dataframe(reports, use_container_width=True, hide_index=True)
    selected = st.selectbox("Report", [item["report_id"] for item in reports])
    report = api("GET", f"/reports/{selected}")
    payload = report["report_json"]
    st.caption(
        f"Scope: {payload.get('report_type')} | Sample: {payload.get('sample_size')} | "
        f"Profile version: {payload.get('profile_version')}"
    )
    st.markdown(payload.get("interpretation", ""))
    with st.expander("Calculated statistics"):
        st.json(payload)
    with st.expander("Analysis limitations"):
        st.write(payload.get("limitations", []))


def show_market_insights() -> None:
    st.header("Market Insights")
    reports = api("GET", "/reports")
    report = next((item for item in reports if item["report_type"] == "all_jobs"), None)
    if not report:
        st.info("Market insights appear after the first run.")
        return
    payload = api("GET", f"/reports/{report['report_id']}")["report_json"]
    left, right = st.columns(2)
    left.subheader("Most requested skills")
    left.bar_chart(payload.get("skill_frequency", {}))
    right.subheader("Application pipeline")
    right.bar_chart(payload.get("application_pipeline", {}))
    first, second = st.columns(2)
    first.subheader("Geography")
    first.bar_chart(payload.get("geographic_distribution", {}))
    second.subheader("Cloud platforms")
    second.bar_chart(payload.get("cloud_platform_distribution", {}))
    st.subheader("Advertisement source languages")
    st.bar_chart(payload.get("advertisement_source_languages", {}))
    st.subheader("Skills by import period")
    for period, counts in payload.get("skill_frequency_by_import_period", {}).items():
        with st.expander(period):
            st.bar_chart(counts)
    st.subheader("Recent report interpretation")
    st.markdown(payload.get("interpretation", ""))


def show_application_tracker() -> None:
    st.header("Application Tracker")
    jobs = api("GET", "/jobs", params={"limit": 1000})
    applications = [
        {
            "Company": item.get("company"),
            "Role": item.get("title"),
            "Status": item.get("application_status"),
            "Job ID": item["job_id"],
        }
        for item in jobs
        if item.get("application_status") != "not_applied"
    ]
    if not applications:
        st.info("No applications are being tracked yet.")
    else:
        st.dataframe(applications, use_container_width=True, hide_index=True)
    show_job_details()


def show_settings_and_runs() -> None:
    st.header("Settings and Runs")
    status = api("GET", "/status")
    first, second, third = st.columns(3)
    first.metric("Analyzed jobs", status["job_count"])
    second.metric("Current model", status["model_id"])
    third.metric("Model loaded", "Yes" if status["model_loaded"] else "No")
    st.caption(f"Database: {status['database_path']}")

    st.subheader("Career profile")
    profile_data = api("GET", "/profile")
    uploaded = st.file_uploader("Import CV", type=["txt", "md", "pdf", "docx"])
    if uploaded is not None and st.button("Extract CV profile"):
        payload = {
            "filename": uploaded.name,
            "content_base64": base64.b64encode(uploaded.getvalue()).decode("ascii"),
        }
        with st.spinner("Extracting CV locally. This may take several minutes."):
            result = api("POST", "/profile/cv", json=payload)
        st.success(result["message"])
        st.session_state["profile_json"] = json.dumps(result["profile"], indent=2)
    profile_json = st.text_area(
        "Review or edit the full profile JSON",
        value=st.session_state.get(
            "profile_json", json.dumps(profile_data["profile"], indent=2)
        ),
        height=360,
    )
    if st.button("Save profile"):
        try:
            profile = json.loads(profile_json)
        except json.JSONDecodeError as error:
            st.error(f"Invalid JSON: {error}")
            return
        result = api("PUT", "/profile", json=profile)
        st.session_state["profile_json"] = json.dumps(result["profile"], indent=2)
        st.success(f"Saved profile version {result['profile_version']}.")

    st.subheader("Model, research, and scoring configuration")
    settings = api("GET", "/settings")
    settings_text = st.text_area(
        "Edit the settings JSON",
        value=json.dumps(settings, indent=2),
        height=320,
        key="settings_json",
    )
    if st.button("Save settings"):
        try:
            parsed = json.loads(settings_text)
            api(
                "PUT",
                "/settings",
                json={
                    "model": parsed["model"],
                    "research": parsed["research"],
                    "scoring_weights": parsed["scoring_weights"],
                    "input_directory": parsed["input_directory"],
                },
            )
        except (json.JSONDecodeError, KeyError) as error:
            st.error(f"Invalid settings: {error}")
        else:
            st.success("Saved settings. Changes apply to the next run.")
            st.rerun()

    st.subheader("Run analysis")
    input_directory = st.text_input("Job directory", value=settings["input_directory"])
    force = st.checkbox("Reprocess unchanged job files", value=False)
    if st.button("Run Analysis", type="primary"):
        result = api(
            "POST",
            "/runs",
            json={"input_directory": input_directory, "force": force},
        )
        st.success(f"Analysis started: {result['run_id']}")
    if status.get("active_run_id"):
        st.info(f"Run in progress: {status['active_run_id']}")

    st.subheader("Run history and errors")
    runs = api("GET", "/runs")
    st.dataframe(runs, use_container_width=True, hide_index=True)
    if runs:
        selected_run = st.selectbox("Inspect run", [item["run_id"] for item in runs])
        run = api("GET", f"/runs/{selected_run}")
        if run.get("errors"):
            st.json(run["errors"])

    st.subheader("Database backup and restore")
    backup_name = st.text_input("Backup name (optional)")
    if st.button("Create database backup"):
        result = api("POST", "/backup", params={"name": backup_name or None})
        st.success(f"Backup created at {result['path']}")
    restore_name = st.text_input("Restore backup name from the backups directory")
    confirm_restore = st.checkbox("Confirm replacement of the current database")
    if st.button("Restore database backup"):
        if not confirm_restore:
            st.warning("Confirm the database replacement before restoring.")
        else:
            result = api("POST", "/backup/restore", json={"backup_name": restore_name})
            st.success(f"Restored {result['backup_name']}.")


if __name__ == "__main__":
    main()
