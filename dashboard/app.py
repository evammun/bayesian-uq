"""
Streamlit dashboard for Bayesian UQ v2 (logprob-based) results.

Aesthetic: "Fountain Pen in a Lab Coat" — Playfair Display for titles,
Inter for body text, teal/rose palette on warm off-white.

Tabs:
  1. Progress — per-run status cards
  2. Probability Distributions — confidence, agreement, mean probs
  3. Condition Comparison — accuracy by subject, AUROC
  4. Effect Analysis — factorial main effects
  5. Question Explorer — drill into individual questions
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

TEAL = "#2D7F83"
DEEP_BLUE = "#1E4D8A"
DARK_TEAL = "#2F555A"
CHARCOAL = "#1A2F32"
SLATE = "#6B7280"
ROSE = "#B85C5C"
GOLD = "#EEB127"
BG = "#FDFCFB"
TEXT = "#2C3E50"
GRAY_MID = "#6B7280"
GRAY_LIGHT = "#8B95A1"
GRID = "#E8E4E0"
BORDER = "#E5E0DB"
RUN_COLORS = [TEAL, DEEP_BLUE, DARK_TEAL, CHARCOAL, SLATE, "#3A8A8F"]
CHOICE_COLORS = [TEAL, DEEP_BLUE, "#C9A227", ROSE]  # A, B, C, D
ANSWER_LETTERS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bayesian UQ v2", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #2C3E50; }
h1 { font-family: 'Playfair Display', serif !important; font-weight: 400 !important; }
h2, h3, h4 { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; }
.stApp { background-color: #FDFCFB; }
[data-testid="stMetric"] { background: transparent; border: 1px solid #E5E0DB;
    border-radius: 8px; padding: 16px; }
[data-testid="stMetricLabel"] { font-size: 13px; text-transform: uppercase;
    letter-spacing: 0.04em; color: #8B95A1; }
[data-testid="stMetricValue"] { font-size: 28px; font-weight: 400; }
.stProgress > div > div > div { background-color: #2D7F83; }
[data-testid="stSidebar"] { background-color: #F5F3F1; }
button[data-baseweb="tab"] { font-weight: 400 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #2D7F83 !important; }
footer, #MainMenu, [data-testid="stDeployButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_result_files() -> list[Path]:
    """Scan results/ for JSON files, newest first."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)


@st.cache_data(ttl=15)
def load_result_file(path: str) -> dict | None:
    """Load a result JSON file. Returns None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=300)
def load_question_db() -> dict[str, dict]:
    """Load questions.json keyed by question_id."""
    if not QUESTIONS_PATH.exists():
        return {}
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_id"]: q for q in data}


# ---------------------------------------------------------------------------
# Run name formatting
# ---------------------------------------------------------------------------

def _parse_model_str(cfg: dict) -> str:
    """Human-readable model string from config."""
    model = cfg.get("model", "unknown")
    name = model.split(":")[0] if ":" in model else model
    family = name.capitalize()
    size = ""
    m = re.search(r"(\d+)[bB]", model)
    if m:
        size = f" {m.group(1)}B"
    quant = ""
    m = re.search(r"(q\d+)", model, re.IGNORECASE)
    if m:
        quant = f" {m.group(1).upper()}"
    return f"{family}{size}{quant}"


def format_run_name(cfg: dict) -> str:
    """Human-readable run label: 'Qwen3 8B Q4 · direct · shuffle on · para on · 100q'"""
    parts = [_parse_model_str(cfg)]
    pm = cfg.get("prompt_mode", "direct")
    if pm != "direct":
        parts.append("CoT" if pm == "cot" else "CoT-struct")
    else:
        parts.append("direct")
    if cfg.get("think", False):
        parts.append("think on")
    parts.append("shuffle on" if cfg.get("shuffle_choices", True) else "shuffle off")
    parts.append("para on" if cfg.get("use_paraphrases", True) else "para off")
    mq = cfg.get("max_questions")
    if mq:
        parts.append(f"{mq}q")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Subject categories
# ---------------------------------------------------------------------------

SUBJECT_CATEGORIES = {
    "STEM": ["math", "physics", "chemistry", "biology", "computer_science",
             "engineering", "astronomy", "machine_learning", "statistics",
             "algebra", "conceptual_physics", "electrical_engineering",
             "abstract_algebra", "college_mathematics", "high_school_mathematics",
             "elementary_mathematics", "college_physics", "high_school_physics",
             "high_school_chemistry", "college_chemistry", "college_biology",
             "high_school_biology", "high_school_computer_science",
             "college_computer_science", "high_school_statistics"],
    "Medical": ["anatomy", "clinical_knowledge", "medicine", "nutrition",
                "virology", "human_aging", "medical_genetics"],
    "Professional": ["professional", "accounting", "marketing", "management", "law"],
    "Social Science": ["economics", "econometrics", "geography", "government",
                       "politics", "sociology", "psychology", "public_relations",
                       "business_ethics", "international_law", "jurisprudence",
                       "human_sexuality", "us_foreign_policy", "security_studies",
                       "computer_security", "microeconomics", "macroeconomics",
                       "high_school_macroeconomics", "high_school_microeconomics",
                       "high_school_government_and_politics", "high_school_geography",
                       "high_school_psychology"],
    "Humanities": ["history", "philosophy", "world_religions", "prehistory",
                   "logical_fallacies", "formal_logic", "moral",
                   "high_school_european_history", "high_school_us_history",
                   "high_school_world_history", "philosophy", "moral_scenarios",
                   "moral_disputes"],
    "Other": [],
}


def get_category(subject: str) -> str:
    """Map a subject string to one of 6 broad categories."""
    s = subject.lower().replace(" ", "_")
    for cat, keywords in SUBJECT_CATEGORIES.items():
        if cat == "Other":
            continue
        for kw in keywords:
            if kw in s:
                return cat
    return "Other"


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_dataframe(run_data: dict) -> pd.DataFrame:
    """Build a DataFrame from a run's question_results."""
    qr_list = run_data.get("question_results", [])
    if not qr_list:
        return pd.DataFrame()

    rows = []
    for qr in qr_list:
        mean_probs = qr.get("mean_probs", [])
        num_queries = qr.get("num_queries", 0)

        # Per-query agreement: fraction where argmax matches final_answer
        final_ans = qr.get("final_answer", 0)
        query_log = qr.get("query_log", [])
        if num_queries > 1:
            agree_count = sum(
                1 for ql in query_log
                if ql.get("canonical_answer") == final_ans
            )
            agreement = agree_count / num_queries
        else:
            agreement = None  # meaningless with 1 query

        # Confidence = max(mean_probs)
        confidence = max(mean_probs) if mean_probs else 0.0

        rows.append({
            "question_id": qr["question_id"],
            "num_queries": num_queries,
            "final_answer": final_ans,
            "correct": qr.get("correct"),
            "mean_probs": mean_probs,
            "confidence": confidence,
            "agreement": agreement,
        })

    df = pd.DataFrame(rows)

    # Add subject from question DB
    qdb = load_question_db()
    df["subject"] = df["question_id"].map(lambda qid: qdb.get(qid, {}).get("subject", "unknown"))
    df["category"] = df["subject"].map(get_category)

    return df


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_auroc(df: pd.DataFrame) -> float | None:
    """AUROC using max(mean_probs) as score, correct as label."""
    valid = df[df["correct"].notna()].copy()
    if len(valid) < 5:
        return None
    y = valid["correct"].astype(int)
    if y.nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y, valid["confidence"]))
    except Exception:
        return None


def compute_timing(run_data: dict, file_path: str | None = None) -> dict:
    """Compute elapsed time and progress."""
    cfg = run_data.get("config", {})
    total_q = cfg.get("max_questions") or len(run_data.get("question_results", []))
    done_q = len(run_data.get("question_results", []))
    pct = done_q / max(total_q, 1)

    ts = run_data.get("timestamp", "")
    elapsed_str = ""
    remaining_str = ""
    if ts and file_path:
        try:
            from datetime import datetime, timezone
            start = datetime.fromisoformat(ts)
            mtime = Path(file_path).stat().st_mtime
            end = datetime.fromtimestamp(mtime, tz=timezone.utc)
            elapsed = (end - start).total_seconds()
            elapsed_str = _fmt_sec(elapsed)
            if 0 < pct < 1:
                remaining = elapsed / pct * (1 - pct)
                remaining_str = _fmt_sec(remaining)
            elif pct >= 1:
                remaining_str = "Done"
        except Exception:
            pass

    return {
        "total_q": total_q,
        "done_q": done_q,
        "pct": pct,
        "elapsed_str": elapsed_str,
        "remaining_str": remaining_str,
    }


def _fmt_sec(s: float) -> str:
    """Format seconds as 'Xh Ym Zs'."""
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def _fig_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply standard layout to a Plotly figure."""
    fig.update_layout(
        title=title,
        title_font=dict(family="Inter", size=16, color=TEXT),
        font=dict(family="Inter", size=12, color=TEXT),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("# Bayesian UQ v2")

result_files = get_result_files()
if not result_files:
    st.sidebar.warning("No result files found in results/")
    st.stop()

# Auto-refresh toggle
auto_refresh = st.sidebar.toggle("Auto-refresh (30s)", value=False)
if auto_refresh:
    time.sleep(0.1)
    st.cache_data.clear()

# Build label → path mapping
label_to_path: dict[str, Path] = {}
for fp in result_files:
    data = load_result_file(str(fp))
    if data is None:
        continue
    cfg = data.get("config", {})
    label = format_run_name(cfg)
    # Disambiguate if needed
    if label in label_to_path:
        label = f"{label} ({fp.stem})"
    label_to_path[label] = fp

selected_labels = st.sidebar.multiselect(
    "Select runs", list(label_to_path.keys()),
    default=list(label_to_path.keys())[:1] if label_to_path else [],
)

# Show metadata for selected runs
for label in selected_labels:
    fp = label_to_path[label]
    data = load_result_file(str(fp))
    if data is None:
        continue
    cfg = data.get("config", {})
    n = len(data.get("question_results", []))
    mq = cfg.get("max_questions", n)
    st.sidebar.caption(
        f"**{label}**  \n"
        f"Model: {cfg.get('model', '?')} · Prompt: {cfg.get('prompt_mode', '?')} · "
        f"Shuffle: {'on' if cfg.get('shuffle_choices') else 'off'} · "
        f"Para: {'on' if cfg.get('use_paraphrases') else 'off'}  \n"
        f"Questions: {n}/{mq}"
    )

# Load selected runs
runs: list[dict] = []
for label in selected_labels:
    fp = label_to_path[label]
    data = load_result_file(str(fp))
    if data is None:
        continue
    df = build_dataframe(data)
    runs.append({
        "label": label,
        "path": fp,
        "config": data.get("config", {}),
        "raw": data,
        "df": df,
        "question_results": data.get("question_results", []),
    })

if not runs:
    st.info("Select at least one run to view results.")
    st.stop()

# Auto-refresh
if auto_refresh:
    st.rerun()


# ---------------------------------------------------------------------------
# Tab 1: Progress
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Progress",
    "Probability Distributions",
    "Condition Comparison",
    "Effect Analysis",
    "Question Explorer",
])

with tab1:
    st.markdown("# Progress")
    for run in runs:
        timing = compute_timing(run["raw"], str(run["path"]))
        df = run["df"]
        valid = df[df["correct"].notna()]
        acc = valid["correct"].mean() if len(valid) > 0 else 0.0

        st.markdown(f"### {run['label']}")
        cols = st.columns(5)
        cols[0].metric("Questions", f"{timing['done_q']}/{timing['total_q']}")
        cols[1].metric("Accuracy", f"{acc:.1%}" if len(valid) > 0 else "—")
        cols[2].metric("Avg Queries", f"{df['num_queries'].mean():.1f}" if len(df) > 0 else "—")
        cols[3].metric("Elapsed", timing["elapsed_str"] or "—")
        cols[4].metric("Remaining", timing["remaining_str"] or "—")

        st.progress(timing["pct"])
        st.divider()


# ---------------------------------------------------------------------------
# Tab 2: Probability Distributions
# ---------------------------------------------------------------------------

with tab2:
    st.markdown("# What does the model actually believe?")

    # Section 2a: Per-query confidence
    st.markdown("### Per-query confidence (max probability)")
    st.caption("How peaked are the model's per-query distributions? Correct answers should have higher peak probabilities.")

    for run in runs:
        qrs = run["question_results"]
        conf_correct = []
        conf_incorrect = []
        for qr in qrs:
            is_correct = qr.get("correct")
            for ql in qr.get("query_log", []):
                probs = ql.get("canonical_probs", [])
                if probs:
                    peak = max(probs)
                    if is_correct is True:
                        conf_correct.append(peak)
                    elif is_correct is False:
                        conf_incorrect.append(peak)

        fig = go.Figure()
        if conf_correct:
            fig.add_trace(go.Histogram(
                x=conf_correct, name="Correct", marker_color=TEAL,
                opacity=0.7, nbinsx=30,
            ))
        if conf_incorrect:
            fig.add_trace(go.Histogram(
                x=conf_incorrect, name="Incorrect", marker_color=ROSE,
                opacity=0.7, nbinsx=30,
            ))
        fig.update_layout(barmode="overlay")
        _fig_layout(fig, title=f"{run['label']} — Per-query max(prob)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Section 2b: Agreement across paraphrases
    st.markdown("### Agreement across paraphrases")
    st.caption("Fraction of queries where argmax matches the final answer. Correct questions should cluster near 1.0.")

    for run in runs:
        df = run["df"]
        df_multi = df[df["agreement"].notna()]
        if df_multi.empty:
            st.caption(f"{run['label']}: Only single-query results — agreement N/A")
            continue

        fig = go.Figure()
        correct_ag = df_multi[df_multi["correct"] == True]["agreement"]
        incorrect_ag = df_multi[df_multi["correct"] == False]["agreement"]
        if len(correct_ag):
            fig.add_trace(go.Histogram(
                x=correct_ag, name="Correct", marker_color=TEAL,
                opacity=0.7, nbinsx=20,
            ))
        if len(incorrect_ag):
            fig.add_trace(go.Histogram(
                x=incorrect_ag, name="Incorrect", marker_color=ROSE,
                opacity=0.7, nbinsx=20,
            ))
        fig.update_layout(barmode="overlay")
        _fig_layout(fig, title=f"{run['label']} — Agreement distribution", height=300)
        fig.update_xaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

        # Agreement range filter for drill-through
        ag_range = st.slider(
            "Filter by agreement", 0.0, 1.0, (0.0, 0.7),
            key=f"ag_slider_{run['label']}",
        )
        filtered = df_multi[
            (df_multi["agreement"] >= ag_range[0]) &
            (df_multi["agreement"] <= ag_range[1])
        ].sort_values("agreement")
        if not filtered.empty:
            st.caption(f"{len(filtered)} questions with agreement in [{ag_range[0]:.2f}, {ag_range[1]:.2f}]")
            display_df = filtered[["question_id", "agreement", "confidence", "correct", "category"]].copy()
            display_df["agreement"] = display_df["agreement"].map(lambda x: f"{x:.2f}")
            display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.3f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Section 2c: Mean probability of winning answer
    st.markdown("### Mean probability of winning answer")
    st.caption("How decisive is the aggregate across paraphrases?")

    for run in runs:
        df = run["df"]
        valid = df[df["correct"].notna()]

        fig = go.Figure()
        correct_conf = valid[valid["correct"] == True]["confidence"]
        incorrect_conf = valid[valid["correct"] == False]["confidence"]
        if len(correct_conf):
            fig.add_trace(go.Histogram(
                x=correct_conf, name="Correct", marker_color=TEAL,
                opacity=0.7, nbinsx=25,
            ))
        if len(incorrect_conf):
            fig.add_trace(go.Histogram(
                x=incorrect_conf, name="Incorrect", marker_color=ROSE,
                opacity=0.7, nbinsx=25,
            ))
        fig.update_layout(barmode="overlay")
        _fig_layout(fig, title=f"{run['label']} — max(mean_probs)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Section 2d: Summary metrics table
    st.markdown("### Summary metrics")
    summary_rows = []
    for run in runs:
        df = run["df"]
        valid = df[df["correct"].notna()]
        correct_df = valid[valid["correct"] == True]
        incorrect_df = valid[valid["correct"] == False]

        multi_q = df[df["agreement"].notna()]
        correct_multi = multi_q[multi_q["correct"] == True]
        incorrect_multi = multi_q[multi_q["correct"] == False]

        summary_rows.append({
            "Run": run["label"],
            "Accuracy": f"{valid['correct'].mean():.1%}" if len(valid) else "—",
            "Avg conf (correct)": f"{correct_df['confidence'].mean():.3f}" if len(correct_df) else "—",
            "Avg conf (incorrect)": f"{incorrect_df['confidence'].mean():.3f}" if len(incorrect_df) else "—",
            "Avg agreement (correct)": f"{correct_multi['agreement'].mean():.3f}" if len(correct_multi) else "N/A",
            "Avg agreement (incorrect)": f"{incorrect_multi['agreement'].mean():.3f}" if len(incorrect_multi) else "N/A",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Condition Comparison
# ---------------------------------------------------------------------------

with tab3:
    st.markdown("# Condition Comparison")

    if len(runs) < 2:
        st.info("Select two or more runs to compare conditions.")
    else:
        # Accuracy heatmap by category × run
        st.markdown("### Accuracy by subject category")
        categories = ["STEM", "Medical", "Humanities", "Social Science", "Professional", "Other"]
        heat_data = []
        for run in runs:
            df = run["df"]
            valid = df[df["correct"].notna()]
            row = {}
            for cat in categories:
                cat_df = valid[valid["category"] == cat]
                row[cat] = cat_df["correct"].mean() if len(cat_df) > 0 else None
            heat_data.append(row)

        heat_df = pd.DataFrame(heat_data, index=[r["label"] for r in runs])
        z = heat_df.values.astype(float)
        text = [[f"{v:.0%}" if not np.isnan(v) else "—" for v in row] for row in z]

        fig = go.Figure(data=go.Heatmap(
            z=z, x=categories, y=[r["label"] for r in runs],
            text=text, texttemplate="%{text}", textfont=dict(color="white", size=13),
            colorscale=[[0, ROSE], [0.5, SLATE], [1, TEAL]],
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(title="Accuracy"),
        ))
        _fig_layout(fig, height=max(200, 60 * len(runs) + 80))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        # Key metrics table
        st.markdown("### Key metrics")
        metric_rows = []
        for run in runs:
            df = run["df"]
            valid = df[df["correct"].notna()]
            auroc = compute_auroc(df)

            multi = df[df["agreement"].notna()]
            correct_multi = multi[multi["correct"] == True]
            incorrect_multi = multi[multi["correct"] == False]

            metric_rows.append({
                "Run": run["label"],
                "Accuracy": f"{valid['correct'].mean():.1%}" if len(valid) else "—",
                "AUROC": f"{auroc:.3f}" if auroc is not None else "—",
                "Avg agree (correct)": f"{correct_multi['agreement'].mean():.3f}" if len(correct_multi) else "N/A",
                "Avg agree (incorrect)": f"{incorrect_multi['agreement'].mean():.3f}" if len(incorrect_multi) else "N/A",
            })

        if metric_rows:
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4: Effect Analysis
# ---------------------------------------------------------------------------

with tab4:
    st.markdown("# Effect Analysis")

    if len(runs) < 2:
        st.info("Select two or more runs to analyse effects.")
    else:
        # Extract condition flags from each run
        run_flags = []
        for run in runs:
            cfg = run["config"]
            df = run["df"]
            valid = df[df["correct"].notna()]
            acc = valid["correct"].mean() if len(valid) else None
            auroc = compute_auroc(df)
            run_flags.append({
                "label": run["label"],
                "shuffle": cfg.get("shuffle_choices", True),
                "para": cfg.get("use_paraphrases", True),
                "prompt_mode": cfg.get("prompt_mode", "direct"),
                "think": cfg.get("think", False),
                "accuracy": acc,
                "auroc": auroc,
            })

        flags_df = pd.DataFrame(run_flags)

        # For each variable, compute main effect if we have matched pairs
        st.markdown("### Main effects on accuracy")
        for var in ["shuffle", "para", "prompt_mode", "think"]:
            unique_vals = flags_df[var].unique()
            if len(unique_vals) < 2:
                continue

            st.markdown(f"**{var}**")
            effect_rows = []
            for val in unique_vals:
                subset = flags_df[flags_df[var] == val]
                avg_acc = subset["accuracy"].mean()
                avg_auroc = subset["auroc"].dropna().mean() if subset["auroc"].notna().any() else None
                effect_rows.append({
                    "Value": str(val),
                    "Runs": len(subset),
                    "Avg accuracy": f"{avg_acc:.1%}" if avg_acc is not None else "—",
                    "Avg AUROC": f"{avg_auroc:.3f}" if avg_auroc is not None else "—",
                })
            st.dataframe(pd.DataFrame(effect_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5: Question Explorer
# ---------------------------------------------------------------------------

with tab5:
    st.markdown("# Question Explorer")

    qdb = load_question_db()
    if not qdb:
        st.warning("Could not load question database.")
        st.stop()

    # Build question options from all selected runs
    all_qids = set()
    for run in runs:
        for qr in run["question_results"]:
            all_qids.add(qr["question_id"])

    if not all_qids:
        st.info("No questions in selected runs.")
        st.stop()

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        # Disagreement filter
        disagree_filter = st.checkbox("Show only disagreements (agreement < 0.7 or runs disagree)", value=False)
    with col_f2:
        # Category filter
        all_cats = sorted({get_category(qdb.get(qid, {}).get("subject", "")) for qid in all_qids})
        cat_filter = st.multiselect("Filter by category", all_cats, default=all_cats)

    # Apply filters
    filtered_qids = []
    for qid in sorted(all_qids):
        q_info = qdb.get(qid, {})
        cat = get_category(q_info.get("subject", ""))
        if cat not in cat_filter:
            continue

        if disagree_filter:
            # Check if any run has low agreement or runs disagree on correctness
            show = False
            answers = set()
            for run in runs:
                df = run["df"]
                row = df[df["question_id"] == qid]
                if row.empty:
                    continue
                row = row.iloc[0]
                if row["agreement"] is not None and row["agreement"] < 0.7:
                    show = True
                answers.add(row["correct"])
            if len(answers) > 1:
                show = True
            if not show:
                continue

        text_preview = q_info.get("question_text", "")[:80]
        filtered_qids.append((qid, f"{qid} — {text_preview}"))

    if not filtered_qids:
        st.info("No questions match the current filters.")
        st.stop()

    selected_q_label = st.selectbox("Select question", [label for _, label in filtered_qids])
    selected_qid = selected_q_label.split(" — ")[0]

    # Display question info
    q_info = qdb.get(selected_qid, {})
    st.markdown(f"**Subject:** {q_info.get('subject', '?')} · **Category:** {get_category(q_info.get('subject', ''))}")
    st.markdown(f"**Question:** {q_info.get('question_text', '?')}")

    # Show choices with correct highlighted
    choices = q_info.get("choices", [])
    correct_idx = q_info.get("correct_answer")
    for i, choice in enumerate(choices):
        letter = ANSWER_LETTERS[i]
        if i == correct_idx:
            st.markdown(f"**{letter})** <span style='color:{TEAL}; font-weight:500;'>{choice} ✓</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{letter})** {choice}")

    st.divider()

    # Per-run details
    for run in runs:
        qr = None
        for q in run["question_results"]:
            if q["question_id"] == selected_qid:
                qr = q
                break
        if qr is None:
            st.caption(f"{run['label']}: question not in this run")
            continue

        mean_probs = qr.get("mean_probs", [])
        final_ans = qr.get("final_answer", 0)
        correct = qr.get("correct")
        query_log = qr.get("query_log", [])
        num_q = qr.get("num_queries", 0)

        # Agreement
        if num_q > 1:
            agree = sum(1 for ql in query_log if ql.get("canonical_answer") == final_ans) / num_q
            agree_str = f"{agree:.2f}"
        else:
            agree_str = "N/A"

        # Summary line
        prob_str = " ".join(f"{ANSWER_LETTERS[i]}:{p:.3f}" for i, p in enumerate(mean_probs))
        correct_str = "✓ Correct" if correct else ("✗ Incorrect" if correct is False else "—")
        correct_color = TEAL if correct else (ROSE if correct is False else GRAY_MID)

        st.markdown(
            f"### {run['label']}\n"
            f"Answer: **{ANSWER_LETTERS[final_ans]}** · "
            f"<span style='color:{correct_color};'>{correct_str}</span> · "
            f"Mean probs: {prob_str} · Agreement: {agree_str}",
            unsafe_allow_html=True,
        )

        # Query detail table
        if query_log:
            table_rows = []
            for ql in query_log:
                probs = ql.get("canonical_probs", [])
                prob_display = " ".join(f"{ANSWER_LETTERS[i]}:{p:.2f}" for i, p in enumerate(probs))
                qt = ql.get("query_text", "")[:80]
                table_rows.append({
                    "#": ql.get("query_number", 0),
                    "Para": ql.get("paraphrase_index", -1),
                    "Query text": qt,
                    "Display ans": ql.get("display_answer", "?"),
                    "Canon ans": ANSWER_LETTERS[ql.get("canonical_answer", 0)],
                    "Probs": prob_display,
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Stacked bar chart of probabilities across queries
        if query_log and len(query_log) > 1:
            fig = go.Figure()
            for choice_idx in range(4):
                vals = [
                    ql.get("canonical_probs", [0, 0, 0, 0])[choice_idx]
                    for ql in query_log
                ]
                fig.add_trace(go.Bar(
                    x=list(range(len(query_log))),
                    y=vals,
                    name=ANSWER_LETTERS[choice_idx],
                    marker_color=CHOICE_COLORS[choice_idx],
                ))
            fig.update_layout(barmode="stack")
            _fig_layout(fig, title="Probability distribution per query", height=300)
            fig.update_xaxes(title="Query number", dtick=1)
            fig.update_yaxes(title="Probability", range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
