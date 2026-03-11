"""
Bayesian UQ — Research Dashboard

A presentation-quality dashboard for visualising experiment results.
Five pages: Progress, Posterior Diagnostics, Condition Comparison, Effect Analysis, Question Explorer.

Launch with:
    streamlit run dashboard/app.py
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score
from streamlit_autorefresh import st_autorefresh


# ---------------------------------------------------------------------------
# 1. Page config and constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bayesian UQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"
ANSWER_LETTERS = ["A", "B", "C", "D"]

# Colour palette
TEAL = "#2D7F83"
DEEP_BLUE = "#1E4D8A"
ROSE = "#B85C5C"
GOLD = "#C4A35A"
SLATE = "#5B6770"
BG = "#FDFCFB"
TRANSPARENT = "rgba(0,0,0,0)"
TEXT = "#2C3E50"
GRAY_MID = "#6B7280"
GRAY_LIGHT = "#8B95A1"
GRID = "#E8E4E0"
BORDER = "#E5E0DB"

# Colours for multi-run overlays (up to 6 runs)
RUN_COLORS = [TEAL, DEEP_BLUE, ROSE, GOLD, SLATE, "#5A9F6E"]

# Stacked area chart colours for answer choices A/B/C/D
CHOICE_COLORS = [TEAL, DEEP_BLUE, GOLD, ROSE]


# ---------------------------------------------------------------------------
# 2. CSS injection — Google Fonts + aesthetic overrides
# ---------------------------------------------------------------------------

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* Global font and background */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 400;
    color: #2C3E50 !important;
    background-color: #FDFCFB !important;
}

/* Page titles only — Playfair Display */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 400 !important;
    color: #2C3E50 !important;
}

/* All subheadings — Inter medium, never bold */
h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #2C3E50 !important;
}

/* Metric cards — consistent sizing, thin border */
div[data-testid="metric-container"] {
    background: transparent !important;
    border: 1px solid #E5E0DB !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
}
div[data-testid="metric-container"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #8B95A1 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 28px !important;
    font-weight: 400 !important;
    color: #2C3E50 !important;
}

/* Progress bar — teal fill */
div[data-testid="stProgress"] > div > div > div {
    background-color: #2D7F83 !important;
}
div[data-testid="stProgress"] > div > div {
    background-color: #2D7F83 !important;
}
.stProgress > div > div > div > div {
    background-color: #2D7F83 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F5F3F1 !important;
}
section[data-testid="stSidebar"] h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}

/* Tab labels */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    color: #6B7280 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2D7F83 !important;
    border-bottom-color: #2D7F83 !important;
}

/* Dataframe / table styling */
.stDataFrame {
    border: 1px solid #E5E0DB !important;
    border-radius: 6px !important;
}
.stDataFrame th {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #2C3E50 !important;
}
.stDataFrame td {
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
}

/* Hide deploy button and footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. Plotly theme helper
# ---------------------------------------------------------------------------

def apply_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the project's visual theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        title_font=dict(family="Inter, sans-serif", size=13, color=GRAY_MID, weight=500),
        margin=dict(l=12, r=12, t=36, b=48),
        legend=dict(
            bgcolor=TRANSPARENT,
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=12, family="Inter, sans-serif", color=GRAY_MID),
        ),
        xaxis=dict(
            gridcolor=GRID,
            linecolor=BORDER,
            tickfont=dict(size=11, color=GRAY_LIGHT, family="Inter, sans-serif"),
            title_font=dict(size=12, color=GRAY_MID, family="Inter, sans-serif"),
        ),
        yaxis=dict(
            gridcolor=GRID,
            linecolor=BORDER,
            tickfont=dict(size=11, color=GRAY_LIGHT, family="Inter, sans-serif"),
            title_font=dict(size=12, color=GRAY_MID, family="Inter, sans-serif"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Data loading (cached)
# ---------------------------------------------------------------------------

def get_result_files() -> list[Path]:
    """Scan results/ for JSON files, newest first by modification time."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        RESULTS_DIR.glob("*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )


@st.cache_data(ttl=15)
def load_result_file(path: str) -> dict | None:
    """Load and parse a result JSON file. Returns None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=300)
def load_question_db() -> dict[str, dict]:
    """Load full question records from questions.json, keyed by question_id."""
    if not QUESTIONS_PATH.exists():
        return {}
    try:
        with open(QUESTIONS_PATH, encoding="utf-8") as f:
            questions = json.load(f)
        return {q["question_id"]: q for q in questions}
    except (json.JSONDecodeError, KeyError):
        return {}


def _subject_from_id(qid: str) -> str:
    """Extract subject from question_id like 'mmlu_redux_abstract_algebra_0042'."""
    if qid.startswith("mmlu_redux_"):
        remainder = qid[len("mmlu_redux_"):]
        parts = remainder.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return "unknown"


def build_dataframe(
    question_results: list[dict],
    question_db: dict[str, dict],
) -> pd.DataFrame:
    """Flatten question results into a one-row-per-question DataFrame."""
    rows = []
    for r in question_results:
        qid = r["question_id"]
        q = question_db.get(qid, {})
        alpha = r.get("final_alpha", [])
        alpha_sum = sum(alpha) if alpha else 1
        concentration = max(alpha) / alpha_sum if alpha else 0
        rows.append({
            "question_id": qid,
            "subject": q.get("subject", _subject_from_id(qid)),
            "queries_used": r["queries_used"],
            "stopped_early": r["stopped_early"],
            "final_exceedance": r["final_exceedance"],
            "final_entropy": r["final_entropy"],
            "final_answer": r["final_answer"],
            "correct": r.get("correct"),
            "final_alpha": alpha,
            "concentration": concentration,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4b. Human-readable run name
# ---------------------------------------------------------------------------

def _parse_model_str(cfg: dict) -> str:
    """Extract a human-readable model name from config, e.g. 'Qwen3 8B Q4'."""
    raw_model = cfg.get("model", "unknown")
    # Extract model family and size from before the colon
    parts = raw_model.split(":")
    family = parts[0].capitalize()
    size_tag = ""
    quant_tag = ""
    if len(parts) > 1:
        detail = parts[1]
        # Extract size (e.g., "8b")
        size_match = re.search(r"(\d+[bB])", detail)
        if size_match:
            size_tag = f" {size_match.group(1).upper()}"
        # Extract quantisation (e.g., "q4")
        quant_match = re.search(r"(q\d+)", detail, re.IGNORECASE)
        if quant_match:
            quant_tag = f" {quant_match.group(1).upper()}"
    return f"{family}{size_tag}{quant_tag}"


def format_run_name(cfg: dict) -> str:
    """Generate a readable label from config fields.

    e.g. 'Qwen3 8B Q4 · think on · shuffle on · para on · 100q'
    """
    model_str = _parse_model_str(cfg)

    # Config flags
    think = "think on" if cfg.get("think") else "think off"
    shuffle = "shuffle on" if cfg.get("shuffle_choices", True) else "shuffle off"
    para = "para on" if cfg.get("use_paraphrases") else "para off"

    tokens = [model_str, think, shuffle, para]

    # Question count
    max_q = cfg.get("max_questions")
    if max_q:
        tokens.append(f"{max_q}q")

    return " \u00b7 ".join(tokens)


def format_short_labels(runs_list: list[dict]) -> dict[str, str]:
    """Build short comparison labels, dropping fields shared by all runs.

    Returns a mapping of full display_name -> shortened label. If all runs
    share the same model and question count, those parts are omitted so
    only the differing variables remain (e.g. 'think off · shuffle on · para off').
    """
    if len(runs_list) < 2:
        return {r["display_name"]: r["display_name"] for r in runs_list}

    configs = [r["config"] for r in runs_list]

    # Check which components are shared across all runs
    models = {_parse_model_str(c) for c in configs}
    max_qs = {c.get("max_questions") for c in configs}
    thinks = {bool(c.get("think")) for c in configs}
    shuffles = {bool(c.get("shuffle_choices", True)) for c in configs}
    paras = {bool(c.get("use_paraphrases")) for c in configs}

    # Build per-run short labels including only differing parts
    result = {}
    for r in runs_list:
        cfg = r["config"]
        tokens = []

        # Include model only if it varies
        if len(models) > 1:
            tokens.append(_parse_model_str(cfg))

        # Always include variable flags; skip fixed ones
        if len(thinks) > 1:
            tokens.append("think on" if cfg.get("think") else "think off")
        if len(shuffles) > 1:
            tokens.append("shuffle on" if cfg.get("shuffle_choices", True) else "shuffle off")
        if len(paras) > 1:
            tokens.append("para on" if cfg.get("use_paraphrases") else "para off")

        # Include question count only if it varies
        if len(max_qs) > 1:
            mq = cfg.get("max_questions")
            if mq:
                tokens.append(f"{mq}q")

        # Fallback: if everything is shared, use the full name
        if not tokens:
            tokens = [r["display_name"]]

        result[r["display_name"]] = " \u00b7 ".join(tokens)

    return result


# ---------------------------------------------------------------------------
# 5. Analytics helpers
# ---------------------------------------------------------------------------

def compute_auroc(df: pd.DataFrame) -> float | None:
    """AUROC using final_exceedance as score, correct as label."""
    valid = df[df["correct"].notna()].copy()
    if len(valid) < 5:
        return None
    y = valid["correct"].astype(int)
    if y.nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y, valid["final_exceedance"]))
    except Exception:
        return None


def compute_ece(df: pd.DataFrame, n_bins: int = 10) -> float | None:
    """Expected Calibration Error — weighted mean |confidence - accuracy| per bin."""
    valid = df[df["correct"].notna()].copy()
    if len(valid) < 10:
        return None
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(valid)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (valid["final_exceedance"] >= lo) & (valid["final_exceedance"] <= hi)
        else:
            mask = (valid["final_exceedance"] >= lo) & (valid["final_exceedance"] < hi)
        bin_df = valid[mask]
        if len(bin_df) == 0:
            continue
        mean_conf = float(bin_df["final_exceedance"].mean())
        mean_acc = float(bin_df["correct"].astype(float).mean())
        ece += (len(bin_df) / total) * abs(mean_conf - mean_acc)
    return round(ece, 4)


def compute_timing(run_data: dict, file_path: str | None = None) -> dict:
    """Compute elapsed time, estimated remaining, and percent complete.

    When the experiment is finished and file_path is provided, uses the
    file's last-modified time as the end timestamp so the elapsed counter
    stops ticking.
    """
    try:
        start_ts = datetime.fromisoformat(run_data["timestamp"])
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=timezone.utc)

        n_done = len(run_data.get("question_results", []))
        max_q = run_data["config"].get("max_questions") or n_done
        is_complete = n_done >= max_q and n_done > 0

        # Use file mtime as end time for completed runs, otherwise now
        if is_complete and file_path:
            end_ts = datetime.fromtimestamp(
                Path(file_path).stat().st_mtime, tz=timezone.utc,
            )
        else:
            end_ts = datetime.now(timezone.utc)

        elapsed_sec = (end_ts - start_ts).total_seconds()

        if n_done == 0:
            return {"elapsed_str": _fmt_sec(elapsed_sec), "remaining_str": "\u2014", "pct_complete": 0.0}

        pct = min(n_done / max_q, 1.0) if max_q > 0 else 1.0

        if is_complete:
            return {"elapsed_str": _fmt_sec(elapsed_sec), "remaining_str": "Done", "pct_complete": 1.0}

        avg_sec_per_q = elapsed_sec / n_done
        remaining_sec = avg_sec_per_q * (max_q - n_done)
        return {
            "elapsed_str": _fmt_sec(elapsed_sec),
            "remaining_str": f"~{_fmt_sec(remaining_sec)}",
            "pct_complete": pct,
        }
    except Exception:
        return {"elapsed_str": "\u2014", "remaining_str": "\u2014", "pct_complete": 0.0}


def _fmt_sec(total_seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs', omitting leading zeroes."""
    total_seconds = max(0.0, total_seconds)
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# 6. Subject category mapping
# ---------------------------------------------------------------------------

_CATEGORY_RULES = [
    ("STEM", [
        "math", "physics", "chemistry", "biology", "computer_science",
        "engineering", "astronomy", "machine_learning", "statistics",
        "algebra", "conceptual_physics", "electrical_engineering",
    ]),
    ("Medical", [
        "anatomy", "clinical_knowledge", "medicine", "nutrition",
        "virology", "human_aging", "medical_genetics",
    ]),
    ("Professional", [
        "professional", "accounting", "marketing", "management", "law",
    ]),
    ("Social Science", [
        "economics", "macroeconomics", "microeconomics", "econometrics",
        "geography", "government", "politics", "sociology", "psychology",
        "public_relations", "business_ethics", "international_law",
        "jurisprudence", "human_sexuality", "us_foreign_policy",
        "security_studies", "computer_security",
    ]),
    ("Humanities", [
        "history", "philosophy", "world_religions", "prehistory",
        "logical_fallacies", "formal_logic", "moral",
    ]),
]


def get_category(subject: str) -> str:
    """Map a subject name to one of 6 broad categories."""
    s = subject.lower()
    for category, keywords in _CATEGORY_RULES:
        for kw in keywords:
            if kw in s:
                return category
    return "Other"


# ---------------------------------------------------------------------------
# 7. Sidebar — run selection and auto-refresh
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Bayesian UQ")

    auto_refresh = st.toggle("Auto-refresh (30s)", value=False, key="auto_refresh_toggle")
    if auto_refresh:
        st_autorefresh(interval=30_000, key="auto_refresh_counter")

    st.markdown("---")

    result_files = get_result_files()
    if not result_files:
        st.warning("No result files in results/. Run an experiment first.")
        st.stop()

    # Build label-to-path mapping using human-readable names
    file_labels: dict[str, Path] = {}
    file_configs: dict[str, dict] = {}  # store config for sidebar display
    for f in result_files:
        d = load_result_file(str(f))
        if d is not None:
            cfg = d.get("config", {})
            label = format_run_name(cfg)
            # Disambiguate if two runs have the same formatted name
            if label in file_labels:
                label = f"{label} ({f.stem})"
            file_labels[label] = f
            file_configs[label] = cfg

    if not file_labels:
        st.warning("No valid result files found.")
        st.stop()

    # Persist multiselect across auto-refreshes via session_state
    if "selected_runs" not in st.session_state:
        st.session_state["selected_runs"] = [list(file_labels.keys())[0]]
    else:
        valid = set(file_labels.keys())
        st.session_state["selected_runs"] = [
            s for s in st.session_state["selected_runs"] if s in valid
        ]

    selected_labels = st.multiselect(
        "Select runs",
        options=list(file_labels.keys()),
        key="selected_runs",
        help="One run for detail view, multiple for comparison.",
    )

    if not selected_labels:
        st.info("Select at least one run.")
        st.stop()

    # Show metadata for each selected run
    st.markdown("---")
    for label in selected_labels:
        d = load_result_file(str(file_labels[label]))
        if d is None:
            continue
        cfg = d["config"]
        n_done = len(d.get("question_results", []))
        max_q = cfg.get("max_questions") or "all"
        para_status = "paraphrases on" if cfg.get("use_paraphrases") else "paraphrases off"
        st.markdown(
            f"<div style='margin-bottom:12px;'>"
            f"<span style='font-size:13px; font-weight:500; color:{TEXT};'>{label}</span><br>"
            f"<span style='font-size:11px; color:{GRAY_LIGHT};'>"
            f"{cfg.get('run_name', '')} \u00b7 {cfg['model']} \u00b7 {n_done}/{max_q} questions"
            f" \u00b7 {para_status}"
            f"</span></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# 8. Load selected runs
# ---------------------------------------------------------------------------

question_db = load_question_db()

runs: list[dict] = []
for label in selected_labels:
    path = file_labels[label]
    d = load_result_file(str(path))
    if d is None:
        continue
    cfg = d["config"]
    qr_list = d.get("question_results", [])
    runs.append({
        "label": label,
        "path": path,
        "display_name": format_run_name(cfg),
        "run_name": cfg["run_name"],
        "config": cfg,
        "raw": d,
        "df": build_dataframe(qr_list, question_db),
        "question_results": qr_list,
    })

if not runs:
    st.error("Could not load any selected result files.")
    st.stop()

multi = len(runs) > 1


# ---------------------------------------------------------------------------
# 9. Tab routing
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab_effects, tab4 = st.tabs([
    "Progress",
    "Posterior Diagnostics",
    "Condition Comparison",
    "Effect Analysis",
    "Question Explorer",
])


# ---------------------------------------------------------------------------
# 10. Page 1 — Progress
# ---------------------------------------------------------------------------

with tab1:
    st.header("Run Progress")

    # Sort newest first so the currently-running experiment appears at the top
    runs_by_recency = sorted(
        runs,
        key=lambda r: Path(r["path"]).stat().st_mtime if r.get("path") else 0,
        reverse=True,
    )

    for idx, run in enumerate(runs_by_recency):
        cfg = run["config"]
        raw = run["raw"]
        df = run["df"]
        timing = compute_timing(raw, file_path=run.get("path"))

        n_done = len(run["question_results"])
        max_q = cfg.get("max_questions") or n_done
        valid = df[df["correct"].notna()]
        n_correct = int((valid["correct"] == True).sum())  # noqa: E712
        acc = n_correct / len(valid) if len(valid) > 0 else None

        # Run name as section heading (formatted)
        st.markdown(f"#### {run['display_name']}")

        # Config summary in small gray text
        st.caption(
            f"{cfg.get('run_name', '')} \u00b7 "
            f"threshold={cfg['confidence_threshold']} \u00b7 "
            f"max queries/q={cfg['max_queries_per_question']}"
        )

        # Metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Questions", f"{n_done} / {max_q}")
        c2.metric("Accuracy", f"{acc:.1%}" if acc is not None else "\u2014")
        c3.metric("Avg queries", f"{df['queries_used'].mean():.1f}" if n_done > 0 else "\u2014")
        c4.metric("Elapsed", timing["elapsed_str"])

        # "Done" in teal, estimates in default colour
        remaining = timing["remaining_str"]
        if remaining == "Done":
            c5.markdown(
                f"<div style='border:1px solid {BORDER}; border-radius:8px; padding:16px 20px;'>"
                f"<span style='font-size:13px; color:{GRAY_LIGHT}; text-transform:uppercase; "
                f"letter-spacing:0.04em;'>Remaining</span><br>"
                f"<span style='font-size:28px; font-weight:400; color:{TEAL};'>Done</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            c5.metric("Remaining", remaining)

        # Teal progress bar
        st.progress(min(max(timing["pct_complete"], 0.0), 1.0))

        # Divider between runs
        if idx < len(runs_by_recency) - 1:
            st.markdown(
                "<hr style='border:none; border-top:1px solid #E5E0DB; margin:24px 0;'>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# 11. Page 2 — Posterior Diagnostics
# ---------------------------------------------------------------------------

def _pct_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    xaxis_title: str,
    nbins: int = 20,
) -> go.Figure:
    """Percentage-within-group histogram, correct vs incorrect overlapping.

    Each group (correct/incorrect) is normalised to 100% separately, so
    they're comparable despite different group sizes.
    """
    fig = go.Figure()
    valid = df[df["correct"].notna()].copy()

    for label, colour, mask in [
        ("Correct", TEAL, valid["correct"] == True),    # noqa: E712
        ("Incorrect", ROSE, valid["correct"] == False),  # noqa: E712
    ]:
        subset = valid[mask]
        if len(subset) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=subset[column],
            histnorm="percent",
            name=label,
            marker_color=colour,
            opacity=0.55,
            nbinsx=nbins,
        ))

    fig.update_layout(
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title="% of group",
        barmode="overlay",
    )
    return fig


with tab2:
    st.header("Posterior Diagnostics")
    st.caption("Does the posterior know what it knows?")

    # --- Section 1: Uncertainty signature scatter plot (hero visualisation) ---
    st.markdown("#### Uncertainty signature")
    st.caption(
        "Each dot is one question. Top-left = fast and decisive. "
        "Bottom-right = slow and uncertain."
    )

    if multi:
        cols = st.columns(len(runs))
        for col, run in zip(cols, runs):
            with col:
                rdf = run["df"]
                valid = rdf[rdf["correct"].notna()].copy()
                fig = go.Figure()
                for label, colour, mask in [
                    ("Correct", TEAL, valid["correct"] == True),      # noqa: E712
                    ("Incorrect", ROSE, valid["correct"] == False),    # noqa: E712
                ]:
                    subset = valid[mask]
                    if len(subset) == 0:
                        continue
                    fig.add_trace(go.Scatter(
                        x=subset["queries_used"],
                        y=subset["concentration"],
                        mode="markers",
                        marker=dict(color=colour, size=8, opacity=0.6),
                        name=label,
                    ))
                fig.update_layout(
                    title_text=run["display_name"],
                    xaxis_title="Queries used",
                    yaxis_title="Concentration ratio",
                    yaxis_range=[0, 1.05],
                )
                st.plotly_chart(apply_theme(fig, height=340), use_container_width=True)
    else:
        rdf = runs[0]["df"]
        valid = rdf[rdf["correct"].notna()].copy()
        fig = go.Figure()
        for label, colour, mask in [
            ("Correct", TEAL, valid["correct"] == True),      # noqa: E712
            ("Incorrect", ROSE, valid["correct"] == False),    # noqa: E712
        ]:
            subset = valid[mask]
            if len(subset) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=subset["queries_used"],
                y=subset["concentration"],
                mode="markers",
                marker=dict(color=colour, size=8, opacity=0.6),
                name=label,
            ))
        fig.update_layout(
            xaxis_title="Queries used",
            yaxis_title="Concentration ratio",
            yaxis_range=[0, 1.05],
        )
        st.plotly_chart(apply_theme(fig, height=400), use_container_width=True)

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Section 2: Convergence speed histograms ---
    st.markdown("#### Convergence speed — queries to stopping")
    st.caption(
        "How many queries the framework needs before stopping. "
        "Faster convergence on correct answers = the uncertainty signal is working."
    )

    if multi:
        cols = st.columns(len(runs))
        for col, run in zip(cols, runs):
            with col:
                fig = _pct_histogram(
                    run["df"], "queries_used", run["display_name"],
                    "Queries used",
                )
                st.plotly_chart(apply_theme(fig, height=260), use_container_width=True)
    else:
        fig = _pct_histogram(
            runs[0]["df"], "queries_used", "",
            "Queries used",
        )
        st.plotly_chart(apply_theme(fig, height=300), use_container_width=True)

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Section 2b: Drill-through by convergence speed ---
    st.markdown("#### Explore questions by convergence speed")

    # Use the first selected run for the drill-through (filters apply to it)
    _drill_run = runs[0]
    _drill_df = _drill_run["df"].copy()
    _drill_qr_map = {qr["question_id"]: qr for qr in _drill_run["question_results"]}

    # Range slider for queries used
    _q_min = int(_drill_df["queries_used"].min()) if len(_drill_df) > 0 else 1
    _q_max = int(_drill_df["queries_used"].max()) if len(_drill_df) > 0 else 100
    _q_range = st.slider(
        "Filter by queries used",
        min_value=_q_min,
        max_value=_q_max,
        value=(_q_min, _q_max),
        key="drill_queries_range",
    )

    # Correctness filter
    _drill_filter = st.radio(
        "Show",
        ["All", "Correct only", "Incorrect only"],
        horizontal=True,
        key="drill_correct_filter",
    )

    # Apply filters
    _filtered = _drill_df[
        (_drill_df["queries_used"] >= _q_range[0])
        & (_drill_df["queries_used"] <= _q_range[1])
    ].copy()
    if _drill_filter == "Correct only":
        _filtered = _filtered[_filtered["correct"] == True]   # noqa: E712
    elif _drill_filter == "Incorrect only":
        _filtered = _filtered[_filtered["correct"] == False]  # noqa: E712

    # Sort by queries used descending (budget-exhausters first)
    _filtered = _filtered.sort_values("queries_used", ascending=False).reset_index(drop=True)

    # Truncate to 20
    _truncated = len(_filtered) > 20
    _display = _filtered.head(20)

    if _truncated:
        st.caption(f"Showing top 20 by queries used (of {len(_filtered)} matching)")
    else:
        st.caption(f"{len(_display)} questions match the filter")

    if len(_display) == 0:
        st.info("No questions match the current filter.")
    else:
        for _, row in _display.iterrows():
            qid = row["question_id"]
            q_info = question_db.get(qid, {})
            q_text = q_info.get("question_text", qid)
            q_preview = q_text[:100] + ("..." if len(q_text) > 100 else "")
            correct_icon = "\u2714" if row["correct"] is True else ("\u2718" if row["correct"] is False else "\u2014")
            ans_letter = ANSWER_LETTERS[row["final_answer"]] if row["final_answer"] < len(ANSWER_LETTERS) else "?"
            alpha = row.get("final_alpha", [])
            alpha_str = "[" + ", ".join(str(int(round(v))) for v in alpha) + "]" if alpha else "\u2014"

            # Summary line for the expander header
            header = (
                f"{correct_icon}  {q_preview}  \u2014  "
                f"{row['queries_used']}q, conc={row['concentration']:.2f}"
            )

            with st.expander(header, expanded=False):
                # Full question text
                st.markdown(
                    f"<p style='font-size:14px; color:{TEXT};'>{q_text}</p>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Subject: {row['subject']}  \u00b7  "
                    f"Category: {get_category(row['subject'])}"
                )

                # Answer choices — correct answer highlighted in teal
                choices = q_info.get("choices", [])
                correct_idx = q_info.get("correct_answer")
                if choices:
                    choice_cols = st.columns(len(choices))
                    for ci, (ccol, ctxt) in enumerate(zip(choice_cols, choices)):
                        letter = ANSWER_LETTERS[ci] if ci < len(ANSWER_LETTERS) else str(ci)
                        if ci == correct_idx:
                            ccol.markdown(
                                f"<div style='border:2px solid {TEAL}; border-radius:6px; "
                                f"padding:8px 12px; background:rgba(45,127,131,0.06);'>"
                                f"<span style='color:{TEAL}; font-weight:500;'>{letter})</span> {ctxt}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            ccol.markdown(
                                f"<div style='border:1px solid {BORDER}; border-radius:6px; "
                                f"padding:8px 12px;'>"
                                f"<span style='color:{GRAY_MID};'>{letter})</span> {ctxt}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                # Detail metrics
                st.markdown(
                    f"<p style='font-size:12px; color:{GRAY_MID}; margin-top:8px;'>"
                    f"Answer: <b>{ans_letter}</b>  \u00b7  "
                    f"Queries: <b>{row['queries_used']}</b>  \u00b7  "
                    f"Concentration: <b>{row['concentration']:.3f}</b>  \u00b7  "
                    f"Alpha: <b>{alpha_str}</b>"
                    f"</p>",
                    unsafe_allow_html=True,
                )

                # Exceedance evolution chart (all selected runs overlaid)
                fig_drill = go.Figure()

                threshold = _drill_run["config"].get("confidence_threshold", 0.95)
                fig_drill.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color=GRAY_LIGHT,
                    annotation_text=f"threshold ({threshold})",
                    annotation_position="top right",
                    annotation_font=dict(size=10, color=GRAY_MID),
                )

                for ri, r in enumerate(runs):
                    qr = next(
                        (x for x in r["question_results"] if x["question_id"] == qid),
                        None,
                    )
                    if qr is None or not qr.get("query_log"):
                        continue
                    qlog = qr["query_log"]
                    fig_drill.add_trace(go.Scatter(
                        x=[q["query_number"] + 1 for q in qlog],
                        y=[q["exceedance_after"] for q in qlog],
                        mode="lines+markers",
                        line=dict(color=RUN_COLORS[ri % len(RUN_COLORS)], width=2),
                        marker=dict(size=5, color=RUN_COLORS[ri % len(RUN_COLORS)]),
                        name=r["display_name"],
                    ))

                fig_drill.update_layout(
                    xaxis_title="Query number",
                    yaxis_title="Exceedance probability",
                    yaxis_range=[0, 1.05],
                )
                st.plotly_chart(
                    apply_theme(fig_drill, height=280),
                    use_container_width=True,
                    key=f"drill_exc_{qid}",
                )

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Section 3: Summary metrics table ---
    st.markdown("#### Summary metrics")

    summary_rows = []
    for run in runs:
        rdf = run["df"]
        valid = rdf[rdf["correct"].notna()]
        correct_mask = valid["correct"] == True   # noqa: E712
        incorrect_mask = valid["correct"] == False  # noqa: E712
        n_correct = int(correct_mask.sum())
        acc = n_correct / len(valid) if len(valid) > 0 else None

        # Average queries split by correctness
        avg_q_correct = float(valid.loc[correct_mask, "queries_used"].mean()) if correct_mask.any() else None
        avg_q_incorrect = float(valid.loc[incorrect_mask, "queries_used"].mean()) if incorrect_mask.any() else None

        # Average concentration split by correctness
        avg_conc_correct = float(valid.loc[correct_mask, "concentration"].mean()) if correct_mask.any() else None
        avg_conc_incorrect = float(valid.loc[incorrect_mask, "concentration"].mean()) if incorrect_mask.any() else None

        summary_rows.append({
            "Run": run["display_name"],
            "Paraphrases": "on" if run["config"].get("use_paraphrases") else "off",
            "Accuracy": acc,
            "AUROC": compute_auroc(rdf),
            "ECE": compute_ece(rdf),
            "Avg queries (correct)": round(avg_q_correct, 1) if avg_q_correct is not None else None,
            "Avg queries (incorrect)": round(avg_q_incorrect, 1) if avg_q_incorrect is not None else None,
            "Avg conc. (correct)": round(avg_conc_correct, 3) if avg_conc_correct is not None else None,
            "Avg conc. (incorrect)": round(avg_conc_incorrect, 3) if avg_conc_incorrect is not None else None,
        })

    summary_df = pd.DataFrame(summary_rows)

    def _highlight_best(s: pd.Series) -> list[str]:
        """Highlight the best value per column in teal."""
        if s.name in ("Run", "Paraphrases"):
            return [""] * len(s)
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.isna().all():
            return [""] * len(s)
        # Lower is better for ECE and query counts; higher is better for the rest
        lower_is_better = ("ECE", "Avg queries (correct)", "Avg queries (incorrect)",
                           "Avg conc. (incorrect)")
        if s.name in lower_is_better:
            best = numeric.min()
        else:
            best = numeric.max()
        return [
            "background-color: rgba(45, 127, 131, 0.12)" if v == best else ""
            for v in numeric
        ]

    styled = (
        summary_df.style
        .format({
            "Accuracy": lambda x: f"{x:.1%}" if pd.notna(x) else "\u2014",
            "AUROC": lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014",
            "ECE": lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014",
            "Avg queries (correct)": lambda x: f"{x:.1f}" if pd.notna(x) else "\u2014",
            "Avg queries (incorrect)": lambda x: f"{x:.1f}" if pd.notna(x) else "\u2014",
            "Avg conc. (correct)": lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014",
            "Avg conc. (incorrect)": lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014",
        })
        .apply(_highlight_best)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Section 4: Posterior shape breakdown ---
    st.markdown("#### Posterior shape breakdown")
    st.caption(
        "Average alpha profile for correct vs incorrect answers. "
        "High concentration = decisive. Low concentration = genuinely confused."
    )

    for run in runs:
        if multi:
            st.markdown(f"**{run['display_name']}**")

        rdf = run["df"]
        valid = rdf[rdf["correct"].notna()].copy()

        shape_rows = []
        for label, mask in [
            ("Correct", valid["correct"] == True),      # noqa: E712
            ("Incorrect", valid["correct"] == False),    # noqa: E712
        ]:
            subset = valid[mask]
            if len(subset) == 0:
                shape_rows.append({
                    "Group": label,
                    "Count": 0,
                    "Avg alpha": "\u2014",
                    "Avg concentration": "\u2014",
                })
                continue

            # Compute mean alpha vector across all questions in this group
            alphas = np.array(subset["final_alpha"].tolist())
            mean_alpha = alphas.mean(axis=0)
            mean_conc = float(subset["concentration"].mean())
            alpha_str = "[" + ", ".join(f"{v:.1f}" for v in mean_alpha) + "]"

            shape_rows.append({
                "Group": label,
                "Count": len(subset),
                "Avg alpha": alpha_str,
                "Avg concentration": f"{mean_conc:.3f}",
            })

        st.dataframe(
            pd.DataFrame(shape_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Section 5: Calibration curve (checkpoint-based) ---
    st.markdown("#### Calibration at query checkpoint")

    # Slider to pick the checkpoint query number
    max_budget = max(
        run["config"].get("max_queries_per_question", 100) for run in runs
    )
    checkpoint = st.slider(
        "Checkpoint query number",
        min_value=1,
        max_value=min(max_budget, 100),
        value=5,
        key="cal_checkpoint",
    )

    fig_cal = go.Figure()

    # Perfect calibration diagonal
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=GRAY_LIGHT, width=1, dash="dot"),
        name="Perfect calibration",
    ))

    for i, run in enumerate(runs):
        # Get exceedance at the checkpoint for each question
        cp_rows = []
        for qr in run["question_results"]:
            qlog = qr.get("query_log", [])
            if not qlog:
                continue
            # Find exceedance at or before the checkpoint (query_number is 0-indexed)
            exc = None
            for entry in qlog:
                if entry["query_number"] < checkpoint:
                    exc = entry["exceedance_after"]
            if exc is None:
                continue
            cp_rows.append({
                "checkpoint_exceedance": exc,
                "correct": qr.get("correct"),
            })

        if not cp_rows:
            continue
        cp_df = pd.DataFrame(cp_rows)
        valid = cp_df[cp_df["correct"].notna()].copy()
        if len(valid) < 5:
            continue

        # Bin by checkpoint exceedance
        n_bins = 8
        bins = np.linspace(0, 1, n_bins + 1)
        cal_rows = []
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            if b == n_bins - 1:
                mask = (valid["checkpoint_exceedance"] >= lo) & (valid["checkpoint_exceedance"] <= hi)
            else:
                mask = (valid["checkpoint_exceedance"] >= lo) & (valid["checkpoint_exceedance"] < hi)
            bin_df = valid[mask]
            if len(bin_df) < 2:
                continue
            cal_rows.append({
                "mean_confidence": float(bin_df["checkpoint_exceedance"].mean()),
                "accuracy": float(bin_df["correct"].astype(float).mean()),
                "count": len(bin_df),
            })

        if not cal_rows:
            continue
        cal_df = pd.DataFrame(cal_rows)
        sizes = cal_df["count"].apply(lambda c: max(5, min(c / 2, 18)))
        fig_cal.add_trace(go.Scatter(
            x=cal_df["mean_confidence"],
            y=cal_df["accuracy"],
            mode="lines+markers",
            marker=dict(size=sizes, color=RUN_COLORS[i % len(RUN_COLORS)]),
            line=dict(color=RUN_COLORS[i % len(RUN_COLORS)], width=2),
            name=run["display_name"],
        ))

    fig_cal.update_layout(
        title_text=f"Calibration at query {checkpoint}",
        xaxis_title=f"Confidence at query {checkpoint}",
        yaxis_title="Actual accuracy",
        xaxis_range=[0, 1.05],
        yaxis_range=[0, 1.05],
    )
    st.plotly_chart(apply_theme(fig_cal, height=400), use_container_width=True)


# ---------------------------------------------------------------------------
# 12. Page 3 — Condition Comparison
# ---------------------------------------------------------------------------

with tab3:
    st.header("Condition Comparison")

    if not multi:
        st.info("Select two or more runs in the sidebar to compare conditions.")
    else:
        # Build shortened labels for comparison (drop shared model/question count)
        short_labels = format_short_labels(runs)

        # --- Heatmap: category accuracy per run ---
        st.markdown("#### Accuracy by subject category")

        heatmap_rows = []
        for run in runs:
            rdf = run["df"].copy()
            rdf["category"] = rdf["subject"].apply(get_category)
            valid = rdf[rdf["correct"].notna()]
            for cat, grp in valid.groupby("category"):
                n_correct = int((grp["correct"] == True).sum())  # noqa: E712
                acc = n_correct / len(grp) if len(grp) > 0 else None
                heatmap_rows.append({
                    "Category": cat,
                    "Run": short_labels[run["display_name"]],
                    "Accuracy": acc,
                })

        if heatmap_rows:
            hm_df = pd.DataFrame(heatmap_rows)
            pivot = hm_df.pivot_table(
                index="Category", columns="Run", values="Accuracy",
            )
            cat_order = ["STEM", "Medical", "Humanities", "Social Science", "Professional", "Other"]
            cat_order = [c for c in cat_order if c in pivot.index]
            pivot = pivot.reindex(cat_order)

            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot.values * 100,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[
                    [0.0, ROSE],
                    [0.5, SLATE],
                    [1.0, TEAL],
                ],
                text=[[f"{v:.0f}%" if pd.notna(v) else "" for v in row] for row in pivot.values * 100],
                texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                hovertemplate="Category: %{y}<br>Run: %{x}<br>Accuracy: %{z:.1f}%<extra></extra>",
                colorbar=dict(title="Acc %", ticksuffix="%"),
            ))
            fig_hm.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(apply_theme(fig_hm, height=350), use_container_width=True)

        st.markdown(
            "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
            unsafe_allow_html=True,
        )

        # --- Key metrics table ---
        st.markdown("#### Key metrics comparison")

        cmp_rows = []
        for run in runs:
            rdf = run["df"]
            valid = rdf[rdf["correct"].notna()]
            correct_mask = valid["correct"] == True   # noqa: E712
            incorrect_mask = valid["correct"] == False  # noqa: E712
            n_correct = int(correct_mask.sum())
            acc = n_correct / len(valid) if len(valid) > 0 else None

            avg_q_correct = float(valid.loc[correct_mask, "queries_used"].mean()) if correct_mask.any() else None
            avg_q_incorrect = float(valid.loc[incorrect_mask, "queries_used"].mean()) if incorrect_mask.any() else None
            avg_conc_correct = float(valid.loc[correct_mask, "concentration"].mean()) if correct_mask.any() else None
            avg_conc_incorrect = float(valid.loc[incorrect_mask, "concentration"].mean()) if incorrect_mask.any() else None

            cmp_rows.append({
                "Run": short_labels[run["display_name"]],
                "Paraphrases": "on" if run["config"].get("use_paraphrases") else "off",
                "Accuracy": acc,
                "AUROC": compute_auroc(rdf),
                "ECE": compute_ece(rdf),
                "Avg queries (correct)": round(avg_q_correct, 1) if avg_q_correct is not None else None,
                "Avg queries (incorrect)": round(avg_q_incorrect, 1) if avg_q_incorrect is not None else None,
                "Avg conc. (correct)": round(avg_conc_correct, 3) if avg_conc_correct is not None else None,
                "Avg conc. (incorrect)": round(avg_conc_incorrect, 3) if avg_conc_incorrect is not None else None,
            })

        cmp_df = pd.DataFrame(cmp_rows)

        def _highlight_best_cmp(s: pd.Series) -> list[str]:
            """Highlight the best value per column in teal."""
            if s.name in ("Run", "Paraphrases"):
                return [""] * len(s)
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.isna().all():
                return [""] * len(s)
            lower_is_better = ("ECE", "Avg queries (correct)", "Avg queries (incorrect)",
                               "Avg conc. (incorrect)")
            if s.name in lower_is_better:
                best = numeric.min()
            else:
                best = numeric.max()
            return [
                "background-color: rgba(45, 127, 131, 0.12)" if v == best else ""
                for v in numeric
            ]

        styled_cmp = (
            cmp_df.style
            .format({
                "Accuracy": lambda x: f"{x:.1%}" if pd.notna(x) else "\u2014",
                "AUROC": lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014",
                "ECE": lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014",
                "Avg queries (correct)": lambda x: f"{x:.1f}" if pd.notna(x) else "\u2014",
                "Avg queries (incorrect)": lambda x: f"{x:.1f}" if pd.notna(x) else "\u2014",
                "Avg conc. (correct)": lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014",
                "Avg conc. (incorrect)": lambda x: f"{x:.3f}" if pd.notna(x) else "\u2014",
            })
            .apply(_highlight_best_cmp)
        )
        st.dataframe(styled_cmp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 12b. Page — Effect Analysis
# ---------------------------------------------------------------------------

def _extract_condition(cfg: dict) -> dict:
    """Extract the three binary condition flags from a run config."""
    return {
        "think": bool(cfg.get("think")),
        "shuffle": bool(cfg.get("shuffle_choices", False)),
        "para": bool(cfg.get("use_paraphrases", False)),
    }


def _condition_label(cond: dict, exclude_var: str | None = None) -> str:
    """Build a human-readable label from condition flags, optionally excluding one variable."""
    parts = []
    if exclude_var != "think":
        parts.append("think on" if cond["think"] else "think off")
    if exclude_var != "shuffle":
        parts.append("shuffle on" if cond["shuffle"] else "shuffle off")
    if exclude_var != "para":
        parts.append("para on" if cond["para"] else "para off")
    return " \u00b7 ".join(parts)


def _compute_run_metrics(run: dict) -> dict:
    """Compute key metrics for a single run."""
    rdf = run["df"]
    valid = rdf[rdf["correct"].notna()]
    correct_mask = valid["correct"] == True  # noqa: E712
    n_correct = int(correct_mask.sum())
    acc = n_correct / len(valid) if len(valid) > 0 else None
    return {
        "accuracy": acc,
        "auroc": compute_auroc(rdf),
        "avg_queries": float(rdf["queries_used"].mean()) if len(rdf) > 0 else None,
        "ece": compute_ece(rdf),
    }


with tab_effects:
    st.header("Effect Analysis")
    st.caption("What does each experimental variable actually do?")

    # --- Index runs by their condition tuple ---
    run_by_condition: dict[tuple, dict] = {}
    for run in runs:
        cond = _extract_condition(run["config"])
        key = (cond["think"], cond["shuffle"], cond["para"])
        run_by_condition[key] = run

    # --- Define the three variables and their matched pairs ---
    VARIABLES = [
        ("think", "Think on vs off"),
        ("shuffle", "Shuffle on vs off"),
        ("para", "Para on vs off"),
    ]
    VAR_INDEX = {"think": 0, "shuffle": 1, "para": 2}

    def _find_matched_pairs(var_name: str) -> list[tuple[dict, dict, dict]]:
        """Find all matched pairs that differ only on var_name.

        Returns a list of (cond_off, run_off, run_on) tuples. cond_off is the
        condition dict with the variable set to False.
        """
        idx = VAR_INDEX[var_name]
        pairs = []
        seen = set()
        for key, run in run_by_condition.items():
            # Build the complementary key (flip the variable)
            flipped = list(key)
            flipped[idx] = not flipped[idx]
            flipped_key = tuple(flipped)

            pair_id = tuple(sorted([key, flipped_key]))
            if pair_id in seen:
                continue
            seen.add(pair_id)

            if flipped_key in run_by_condition:
                # Determine which is off and which is on
                if key[idx]:
                    run_on, run_off = run, run_by_condition[flipped_key]
                    cond_off = _extract_condition(run_by_condition[flipped_key]["config"])
                else:
                    run_off, run_on = run, run_by_condition[flipped_key]
                    cond_off = _extract_condition(run["config"])
                pairs.append((cond_off, run_off, run_on))
        return pairs

    # --- Section 1a: Main effects table ---
    st.markdown("#### Main effects")
    st.caption(
        "Each delta is the average across all matched pairs that differ only on that variable."
    )

    effect_rows = []
    all_pair_data: dict[str, list] = {}  # var_name -> list of per-pair deltas

    for var_name, var_label in VARIABLES:
        pairs = _find_matched_pairs(var_name)
        if not pairs:
            effect_rows.append({
                "Effect": var_label,
                "Pairs": 0,
                "Accuracy \u0394": None,
                "AUROC \u0394": None,
                "Avg Queries \u0394": None,
                "ECE \u0394": None,
            })
            all_pair_data[var_name] = []
            continue

        deltas = {"accuracy": [], "auroc": [], "avg_queries": [], "ece": []}
        pair_details = []

        for cond_off, run_off, run_on in pairs:
            m_off = _compute_run_metrics(run_off)
            m_on = _compute_run_metrics(run_on)

            pair_info = {
                "label": _condition_label(cond_off, exclude_var=var_name),
                "deltas": {},
            }

            for metric in deltas:
                v_off = m_off[metric]
                v_on = m_on[metric]
                if v_off is not None and v_on is not None:
                    d = v_on - v_off
                    deltas[metric].append(d)
                    pair_info["deltas"][metric] = d

            pair_details.append(pair_info)

        all_pair_data[var_name] = pair_details

        def _mean_or_none(lst):
            return sum(lst) / len(lst) if lst else None

        effect_rows.append({
            "Effect": var_label,
            "Pairs": len(pairs),
            "Accuracy \u0394": _mean_or_none(deltas["accuracy"]),
            "AUROC \u0394": _mean_or_none(deltas["auroc"]),
            "Avg Queries \u0394": _mean_or_none(deltas["avg_queries"]),
            "ECE \u0394": _mean_or_none(deltas["ece"]),
        })

    if effect_rows:
        eff_df = pd.DataFrame(effect_rows)

        # Colour function: teal for good, rose for bad
        def _colour_delta(val, higher_is_better: bool) -> str:
            if pd.isna(val) or val is None:
                return ""
            if higher_is_better:
                return f"color: {TEAL}" if val > 0 else (f"color: {ROSE}" if val < 0 else "")
            else:
                return f"color: {TEAL}" if val < 0 else (f"color: {ROSE}" if val > 0 else "")

        def _style_effects(df: pd.DataFrame) -> pd.io.formats.style.Styler:
            styled = df.style.format({
                "Accuracy \u0394": lambda x: f"{x:+.1%}" if pd.notna(x) else "\u2014",
                "AUROC \u0394": lambda x: f"{x:+.3f}" if pd.notna(x) else "\u2014",
                "Avg Queries \u0394": lambda x: f"{x:+.1f}" if pd.notna(x) else "\u2014",
                "ECE \u0394": lambda x: f"{x:+.4f}" if pd.notna(x) else "\u2014",
                "Pairs": lambda x: str(x),
            })
            # Apply colours per column
            styled = styled.map(
                lambda v: _colour_delta(v, higher_is_better=True),
                subset=["Accuracy \u0394", "AUROC \u0394"],
            )
            styled = styled.map(
                lambda v: _colour_delta(v, higher_is_better=False),
                subset=["Avg Queries \u0394", "ECE \u0394"],
            )
            return styled

        st.dataframe(
            _style_effects(eff_df),
            use_container_width=True,
            hide_index=True,
        )

        # Note about missing pairs
        missing_vars = [row["Effect"] for row in effect_rows if row["Pairs"] < 4]
        if missing_vars:
            st.caption(
                f"Some matched pairs are missing for: {', '.join(missing_vars)}. "
                "Load all 8 experimental conditions for a complete analysis."
            )

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Section 1b: Effect consistency (bar charts) ---
    st.markdown("#### Effect consistency")
    st.caption(
        "Are the effects robust? Each bar shows the delta for one matched pair. "
        "If all bars point the same direction, the effect is robust."
    )

    for metric_key, metric_label in [("accuracy", "Accuracy"), ("auroc", "AUROC")]:
        st.markdown(f"**{metric_label} deltas by variable**")
        cols = st.columns(len(VARIABLES))

        for col_idx, (var_name, var_label) in enumerate(VARIABLES):
            with cols[col_idx]:
                pair_details = all_pair_data.get(var_name, [])
                if not pair_details:
                    st.caption(f"{var_label}: no matched pairs")
                    continue

                labels = []
                values = []
                colors = []
                for pd_item in pair_details:
                    d = pd_item["deltas"].get(metric_key)
                    if d is not None:
                        labels.append(pd_item["label"])
                        values.append(d)
                        colors.append(TEAL if d >= 0 else ROSE)

                if not values:
                    st.caption(f"{var_label}: no data")
                    continue

                fig_bar = go.Figure()
                # Format values for display
                if metric_key == "accuracy":
                    text_vals = [f"{v:+.1%}" for v in values]
                else:
                    text_vals = [f"{v:+.3f}" for v in values]

                fig_bar.add_trace(go.Bar(
                    y=labels,
                    x=values,
                    orientation="h",
                    marker_color=colors,
                    text=text_vals,
                    textposition="outside",
                    textfont=dict(size=11),
                ))

                # Add zero line
                fig_bar.add_vline(
                    x=0, line_color=GRAY_LIGHT, line_width=1,
                )

                fig_bar.update_layout(
                    title_text=var_label,
                    showlegend=False,
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(zeroline=False),
                )
                st.plotly_chart(
                    apply_theme(fig_bar, height=max(180, 60 * len(values))),
                    use_container_width=True,
                )

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Section 1c: Interaction spotlight ---
    st.markdown("#### Interaction spotlight")

    interactions_found = []
    for var_name, var_label in VARIABLES:
        pair_details = all_pair_data.get(var_name, [])
        acc_deltas = [
            (pd_item["label"], pd_item["deltas"].get("accuracy"))
            for pd_item in pair_details
            if pd_item["deltas"].get("accuracy") is not None
        ]

        if len(acc_deltas) < 2:
            continue

        # Check if signs differ
        signs = [d >= 0 for _, d in acc_deltas]
        if len(set(signs)) > 1:
            # Find the extremes to describe the interaction
            pos_pairs = [(lbl, d) for lbl, d in acc_deltas if d >= 0]
            neg_pairs = [(lbl, d) for lbl, d in acc_deltas if d < 0]

            best_pos = max(pos_pairs, key=lambda x: x[1]) if pos_pairs else None
            worst_neg = min(neg_pairs, key=lambda x: x[1]) if neg_pairs else None

            # Identify which other variable differs between the extreme pairs
            var_short = var_label.split(" ")[0].lower()
            msg = f"\u26a0\ufe0f **{var_label.split(' on')[0]}** effect on accuracy flips sign: "
            if best_pos:
                msg += f"+{best_pos[1]:.1%} when {best_pos[0]}"
            if worst_neg:
                msg += f", but {worst_neg[1]:+.1%} when {worst_neg[0]}"
            msg += "."
            interactions_found.append(msg)

    if interactions_found:
        for msg in interactions_found:
            st.markdown(msg)
    else:
        st.markdown(
            "All effects are consistent across conditions \u2014 no significant "
            "interactions detected."
        )


# ---------------------------------------------------------------------------
# 13. Page 4 — Question Explorer
# ---------------------------------------------------------------------------

with tab4:
    st.header("Question Explorer")

    # Build the full set of question IDs across all selected runs
    all_qids_set: set[str] = set()
    for run in runs:
        for qr in run["question_results"]:
            all_qids_set.add(qr["question_id"])
    all_qids = sorted(all_qids_set)

    # --- Disagreement filters ---
    st.markdown("#### Find interesting questions")

    filter_disagree = st.checkbox(
        "Show questions where runs disagree on correctness",
        key="filter_disagree",
    )
    filter_all_wrong = st.checkbox(
        "Show questions no run answered correctly",
        key="filter_all_wrong",
    )

    # Precompute per-question correctness across runs
    _qid_correctness: dict[str, list[bool | None]] = {}
    for qid in all_qids:
        results_for_q = []
        for run in runs:
            qr = next((r for r in run["question_results"] if r["question_id"] == qid), None)
            if qr is not None:
                results_for_q.append(qr.get("correct"))
        _qid_correctness[qid] = results_for_q

    # Apply filters
    filtered_qids = all_qids
    if filter_disagree:
        filtered_qids = [
            qid for qid in filtered_qids
            if (any(c is True for c in _qid_correctness[qid])
                and any(c is False for c in _qid_correctness[qid]))
        ]
    if filter_all_wrong:
        filtered_qids = [
            qid for qid in filtered_qids
            if (all(c is False for c in _qid_correctness[qid] if c is not None)
                and any(c is False for c in _qid_correctness[qid]))
        ]

    # Show counts
    if filter_disagree:
        n_disagree = len([
            qid for qid in all_qids
            if (any(c is True for c in _qid_correctness[qid])
                and any(c is False for c in _qid_correctness[qid]))
        ])
        st.caption(f"{n_disagree} questions with disagreement across selected runs")
    if filter_all_wrong:
        n_all_wrong = len([
            qid for qid in all_qids
            if (all(c is False for c in _qid_correctness[qid] if c is not None)
                and any(c is False for c in _qid_correctness[qid]))
        ])
        st.caption(f"{n_all_wrong} questions no run answered correctly")

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # Build dropdown options from filtered list
    dropdown_options = []
    for qid in filtered_qids:
        q = question_db.get(qid, {})
        text_preview = q.get("question_text", "")[:80]
        dropdown_options.append(f"{qid}  \u2014  {text_preview}")

    if not dropdown_options:
        st.info("No questions match the current filters.")
        st.stop()

    selected_option = st.selectbox(
        "Question",
        dropdown_options,
        key="explorer_question",
    )

    # Extract question_id from the selected option
    selected_qid = selected_option.split("  \u2014  ")[0].strip()

    # Display question details
    q_info = question_db.get(selected_qid)
    if q_info:
        st.markdown(
            f"<p style='font-size:15px; font-weight:500; color:{TEXT};'>"
            f"{q_info['question_text']}</p>",
            unsafe_allow_html=True,
        )
        st.caption(f"Subject: {q_info['subject']}  \u00b7  Category: {get_category(q_info['subject'])}")

        # Display choices — correct answer highlighted with teal
        choice_cols = st.columns(len(q_info["choices"]))
        for i, (col, choice_text) in enumerate(zip(choice_cols, q_info["choices"])):
            letter = ANSWER_LETTERS[i] if i < len(ANSWER_LETTERS) else str(i)
            is_correct = q_info.get("correct_answer") == i
            if is_correct:
                col.markdown(
                    f"<div style='border:2px solid {TEAL}; border-radius:6px; "
                    f"padding:8px 12px; background:rgba(45,127,131,0.06);'>"
                    f"<span style='color:{TEAL}; font-weight:500;'>{letter})</span> {choice_text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                col.markdown(
                    f"<div style='border:1px solid {BORDER}; border-radius:6px; "
                    f"padding:8px 12px;'>"
                    f"<span style='color:{GRAY_MID};'>{letter})</span> {choice_text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if q_info.get("correct_answer") is not None:
            st.caption(f"Correct answer: {ANSWER_LETTERS[q_info['correct_answer']]}")
        else:
            st.caption("Broken premise \u2014 no correct answer")

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Exceedance evolution chart (all runs overlaid) ---
    st.markdown("#### Exceedance evolution")

    fig_exc = go.Figure()

    threshold = runs[0]["config"].get("confidence_threshold", 0.95)
    fig_exc.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=GRAY_LIGHT,
        annotation_text=f"threshold ({threshold})",
        annotation_position="top right",
        annotation_font=dict(size=10, color=GRAY_MID),
    )

    for i, run in enumerate(runs):
        qr = next((r for r in run["question_results"] if r["question_id"] == selected_qid), None)
        if qr is None or not qr.get("query_log"):
            continue
        qlog = qr["query_log"]
        fig_exc.add_trace(go.Scatter(
            x=[q["query_number"] + 1 for q in qlog],
            y=[q["exceedance_after"] for q in qlog],
            mode="lines+markers",
            line=dict(color=RUN_COLORS[i % len(RUN_COLORS)], width=2),
            marker=dict(size=6, color=RUN_COLORS[i % len(RUN_COLORS)]),
            name=run["display_name"],
        ))

    fig_exc.update_layout(
        xaxis_title="Query number",
        yaxis_title="Exceedance probability",
        yaxis_range=[0, 1.05],
    )
    st.plotly_chart(apply_theme(fig_exc, height=350), use_container_width=True)

    # --- Comparison table ---
    table_rows = []
    for run in runs:
        qr = next((r for r in run["question_results"] if r["question_id"] == selected_qid), None)
        if qr is None:
            continue
        ans_letter = ANSWER_LETTERS[qr["final_answer"]] if qr["final_answer"] < len(ANSWER_LETTERS) else "?"
        correct_label = {True: "Correct", False: "Wrong", None: "\u2014"}.get(qr.get("correct"), "\u2014")

        # Build final posterior string: "A:1  B:1  C:8  D:1" with max highlighted
        alpha = qr.get("final_alpha", [])
        if alpha:
            max_alpha_idx = int(np.argmax(alpha))
            parts = []
            for k, a_val in enumerate(alpha):
                letter = ANSWER_LETTERS[k] if k < len(ANSWER_LETTERS) else f"O{k}"
                count = int(round(a_val))
                if k == max_alpha_idx:
                    parts.append(f"\u25b8{letter}:{count}")
                else:
                    parts.append(f"{letter}:{count}")
            posterior_str = "  ".join(parts)
        else:
            posterior_str = "\u2014"

        table_rows.append({
            "Run": run["display_name"],
            "Queries": qr["queries_used"],
            "Answer": ans_letter,
            "Exceedance": round(qr["final_exceedance"], 3),
            "Outcome": "Early stop" if qr["stopped_early"] else "Full budget",
            "Final posterior": posterior_str,
            "Correct": correct_label,
        })

    if table_rows:
        tbl_df = pd.DataFrame(table_rows)
        st.dataframe(tbl_df, use_container_width=True, hide_index=True)

    st.markdown(
        "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # --- Alpha evolution (stacked area) ---
    st.markdown("#### Posterior evolution")

    explorer_run_idx = 0
    if multi:
        run_display_names = [r["display_name"] for r in runs]
        explorer_run_name = st.selectbox(
            "Show posterior for",
            run_display_names,
            key="explorer_posterior_run",
        )
        explorer_run_idx = run_display_names.index(explorer_run_name)

    qr = next(
        (r for r in runs[explorer_run_idx]["question_results"]
         if r["question_id"] == selected_qid),
        None,
    )

    if qr and qr.get("query_log"):
        qlog = qr["query_log"]
        num_k = len(qlog[0]["alpha_after"])

        area_rows = []
        prior_prob = 1.0 / num_k
        for k in range(num_k):
            letter = ANSWER_LETTERS[k] if k < len(ANSWER_LETTERS) else f"Opt{k}"
            area_rows.append({"Query": 0, "Answer": letter, "Probability": prior_prob})

        for q in qlog:
            alpha = np.array(q["alpha_after"])
            probs = alpha / alpha.sum()
            for k in range(len(alpha)):
                letter = ANSWER_LETTERS[k] if k < len(ANSWER_LETTERS) else f"Opt{k}"
                area_rows.append({
                    "Query": q["query_number"] + 1,
                    "Answer": letter,
                    "Probability": float(probs[k]),
                })

        area_df = pd.DataFrame(area_rows)

        fig_area = go.Figure()
        for k in range(num_k):
            letter = ANSWER_LETTERS[k] if k < len(ANSWER_LETTERS) else f"Opt{k}"
            subset = area_df[area_df["Answer"] == letter]
            fig_area.add_trace(go.Scatter(
                x=subset["Query"],
                y=subset["Probability"],
                mode="lines",
                name=letter,
                stackgroup="one",
                line=dict(width=0.5, color=CHOICE_COLORS[k % len(CHOICE_COLORS)]),
                fillcolor=CHOICE_COLORS[k % len(CHOICE_COLORS)],
            ))

        fig_area.update_layout(
            xaxis_title="Query number",
            yaxis_title="Posterior probability",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(apply_theme(fig_area, height=350), use_container_width=True)
    else:
        st.caption("No query data available for this question in the selected run.")
