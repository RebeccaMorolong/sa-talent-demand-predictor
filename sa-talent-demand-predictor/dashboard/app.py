"""
SA Talent Demand — Streamlit Dashboard

Run: streamlit run dashboard/app.py
"""

import ast
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SA Talent Demand",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling — clean, data-forward look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=IBM+Plex+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .block-container { padding-top: 1.5rem; }
    h1 { font-weight: 600; letter-spacing: -0.5px; }
    h2 { font-weight: 400; color: #444; border-bottom: 1px solid #eee; padding-bottom: 0.3rem; }

    .stat-card {
        background: #f8f9fa;
        border-left: 4px solid #1a56db;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .stat-value { font-size: 2rem; font-weight: 600; color: #1a56db; }
    .stat-label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }

    .warning-card {
        background: #fff8e6;
        border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading — cached so it doesn't reload on every interaction
# ---------------------------------------------------------------------------

FEATURES_PATH = Path("data/processed/features/job_features.csv")
FORECAST_PATH = Path("data/processed/forecasts/skill_demand_forecast.csv")
AUDIT_PATH = Path("data/processed/audit/degree_rate_by_industry.csv")


@st.cache_data
def load_features() -> pd.DataFrame | None:
    if not FEATURES_PATH.exists():
        return None
    return pd.read_csv(FEATURES_PATH)


@st.cache_data
def load_forecasts() -> pd.DataFrame | None:
    if not FORECAST_PATH.exists():
        return None
    return pd.read_csv(FORECAST_PATH, parse_dates=["ds"])


@st.cache_data
def load_audit() -> pd.DataFrame | None:
    if not AUDIT_PATH.exists():
        return None
    return pd.read_csv(AUDIT_PATH)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🇿🇦 SA Talent Demand")
    st.caption("Hiring on skills, not degrees.")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Skill Demand", "Degree Gatekeeping", "Bias Audit", "Talent Match"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: Stats SA QLFS · CareerJunction · SETA Reports")


# ---------------------------------------------------------------------------
# Helper: no-data placeholder
# ---------------------------------------------------------------------------
def no_data_warning(msg: str = "Run `make scrape && make features` to load data."):
    st.markdown(f'<div class="warning-card">⚠️ No data yet — {msg}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    st.title("South Africa Talent Demand Predictor")
    st.markdown(
        "Investigating whether **degree requirements** in SA job postings "
        "are driving unemployment — and building a fairer talent-matching system."
    )

    df = load_features()
    if df is None:
        no_data_warning()
        st.info("Once data is loaded, you'll see live metrics here.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df):,}</div>
            <div class="stat-label">Job Postings</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        gated_pct = df["requires_degree"].mean() * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{gated_pct:.1f}%</div>
            <div class="stat-label">Degree Gated</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        industries = df["industry"].nunique()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{industries}</div>
            <div class="stat-label">Industries</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        provinces = df[df["province"] != "unknown"]["province"].nunique()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{provinces}</div>
            <div class="stat-label">Provinces</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Postings by Industry")
        industry_counts = df["industry"].value_counts().reset_index()
        industry_counts.columns = ["industry", "count"]
        fig = px.bar(
            industry_counts, x="count", y="industry", orientation="h",
            color="count", color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Postings by Province")
        prov = df[df["province"] != "unknown"]["province"].value_counts().reset_index()
        prov.columns = ["province", "count"]
        fig2 = px.pie(prov, values="count", names="province", hole=0.4,
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
        st.plotly_chart(fig2, use_container_width=True)


def page_skill_demand():
    st.title("Skill Demand Forecast")

    forecasts = load_forecasts()
    df = load_features()

    if df is not None:
        st.subheader("Top Skills Right Now")
        all_skills: list[str] = []
        for entry in df["skills_str"].dropna():
            all_skills.extend([s.strip() for s in entry.split(",") if s.strip()])
        skill_counts = pd.DataFrame(
            Counter(all_skills).most_common(20), columns=["skill", "count"]
        )
        fig = px.bar(
            skill_counts, x="count", y="skill", orientation="h",
            color="count", color_continuous_scale="Teal",
        )
        fig.update_layout(height=500, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    if forecasts is None:
        no_data_warning("Run `make train` to generate forecasts.")
        return

    st.subheader("12-Month Demand Forecast by Skill")
    skills = sorted(forecasts["skill"].unique())
    chosen = st.multiselect("Select skills to compare", skills, default=skills[:4])

    if chosen:
        sub = forecasts[forecasts["skill"].isin(chosen)]
        fig = px.line(
            sub, x="ds", y="yhat", color="skill",
            labels={"ds": "Date", "yhat": "Predicted Mentions", "skill": "Skill"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=400)
        st.plotly_chart(fig, use_container_width=True)


def page_degree_gatekeeping():
    st.title("Degree Gatekeeping Analysis")
    df = load_features()
    if df is None:
        no_data_warning()
        return

    st.subheader("Degree Requirement Rate by Industry")
    rate = (
        df.groupby("industry")["requires_degree"]
        .mean()
        .reset_index()
        .rename(columns={"requires_degree": "degree_rate"})
        .sort_values("degree_rate", ascending=False)
    )
    rate["pct"] = (rate["degree_rate"] * 100).round(1)
    fig = px.bar(
        rate, x="industry", y="pct",
        labels={"pct": "% of postings requiring a degree", "industry": "Industry"},
        color="pct", color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Skill Complexity: Degree vs No-Degree Postings")
    fig2 = px.box(
        df, x=df["requires_degree"].map({0: "No Degree Required", 1: "Degree Required"}),
        y="skill_count",
        labels={"x": "", "skill_count": "Number of Skills Mentioned"},
        color=df["requires_degree"].map({0: "No Degree Required", 1: "Degree Required"}),
        color_discrete_map={"No Degree Required": "#22c55e", "Degree Required": "#ef4444"},
    )
    fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=350)
    st.plotly_chart(fig2, use_container_width=True)

    diff = (
        df[df["requires_degree"] == 1]["skill_count"].mean()
        - df[df["requires_degree"] == 0]["skill_count"].mean()
    )
    if abs(diff) < 1.0:
        st.warning(
            f"⚠️ Degree-gated postings mention only **{diff:+.1f} more skills** on average than "
            "non-degree postings. This suggests degree requirements may not reflect actual job complexity."
        )
    else:
        st.info(f"Degree-gated postings mention **{diff:+.1f} more skills** on average.")


def page_bias_audit():
    st.title("Bias & Fairness Audit")
    audit = load_audit()

    if audit is None:
        no_data_warning("Run `make audit` to generate the bias report.")
        return

    st.subheader("Degree Gatekeeping by Industry")
    audit["degree_rate_pct"] = (audit["degree_rate"] * 100).round(1)
    fig = px.bar(
        audit.sort_values("degree_rate_pct", ascending=True),
        x="degree_rate_pct", y="industry", orientation="h",
        labels={"degree_rate_pct": "Degree requirement rate (%)", "industry": "Industry"},
        color="degree_rate_pct",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=400)
    st.plotly_chart(fig, use_container_width=True)

    df = load_features()
    if df is not None:
        total = len(df)
        gated = df["requires_degree"].sum()
        st.markdown("### Counterfactual Talent Pool")
        st.markdown(
            f"If degree requirements were removed from all **{gated:,}** degree-gated postings "
            f"({gated/total*100:.1f}% of total), those roles would be accessible to candidates "
            "with the right skills — regardless of formal qualifications."
        )

        col1, col2 = st.columns(2)
        col1.metric("Degree-gated postings", f"{gated:,}")
        col2.metric("Skills-only accessible if removed", f"{gated:,}")


def page_talent_match():
    st.title("Talent Matcher")
    st.markdown("Enter a candidate profile to get an unemployment risk score and role recommendations — **no degree filter applied**.")

    col1, col2 = st.columns(2)
    with col1:
        skills_input = st.text_area(
            "Skills (comma-separated)",
            placeholder="python, sql, data analysis, excel",
            height=100,
        )
        province = st.selectbox("Province", [
            "gauteng", "western cape", "kwazulu-natal", "eastern cape",
            "limpopo", "mpumalanga", "north west", "free state", "northern cape",
        ])
    with col2:
        education = st.selectbox(
            "Education Level",
            ["matric", "diploma", "degree", "honours", "masters", "phd", "not_specified"],
        )
        age = st.slider("Age", 18, 65, 28)

    if st.button("Find Opportunities", type="primary"):
        if not skills_input.strip():
            st.error("Please enter at least one skill.")
            return

        skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]
        st.markdown("---")
        st.subheader("Results")

        # Call local API if running, otherwise do a lightweight inline prediction
        import requests as req
        try:
            resp = req.post(
                "http://localhost:8000/match",
                json={"skills": skills, "province": province, "education_level": education, "age": age},
                timeout=3,
            )
            data = resp.json()
        except Exception:
            # Fallback: simple heuristic if API isn't running
            data = {
                "unemployment_risk_score": max(0.1, 0.5 - len(skills) * 0.04),
                "recommended_industries": ["tech", "finance", "logistics"][:3],
                "top_missing_skills": ["communication", "project management", "excel"][:5],
                "degree_not_required_roles": ["Data Capturer", "Junior Analyst", "Admin Coordinator"],
            }

        risk = data["unemployment_risk_score"]
        risk_color = "#22c55e" if risk < 0.3 else "#f59e0b" if risk < 0.6 else "#ef4444"
        st.markdown(
            f'<div class="stat-card"><div class="stat-value" style="color:{risk_color}">'
            f'{risk:.0%}</div><div class="stat-label">Unemployment Risk Score</div></div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Recommended Industries**")
            for ind in data.get("recommended_industries", []):
                st.markdown(f"• {ind.title()}")
        with c2:
            st.markdown("**Skills to Add**")
            for sk in data.get("top_missing_skills", [])[:5]:
                st.markdown(f"• {sk.title()}")
        with c3:
            st.markdown("**Roles (No Degree Filter)**")
            for role in data.get("degree_not_required_roles", []):
                st.markdown(f"• {role}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Overview":
    page_overview()
elif page == "Skill Demand":
    page_skill_demand()
elif page == "Degree Gatekeeping":
    page_degree_gatekeeping()
elif page == "Bias Audit":
    page_bias_audit()
elif page == "Talent Match":
    page_talent_match()
