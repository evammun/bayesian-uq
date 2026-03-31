"""
Streamlit dashboard for Pre-Action UQ (QuALITY-based) results.

Aesthetic: "Fountain Pen in a Lab Coat" — Playfair Display for titles,
Inter for body text, teal/rose palette on warm off-white.

Tabs:
  1. Progress — one-line-per-run status with timing
  2. Uncertainty Distributions — MSP, agreement, entropy overlaid by condition
  3. Condition Comparison — 2×2 accuracy, AUROC, calibration curves
  4. Question Explorer — drill into individual questions
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SIGNALS_CSV = PROJECT_ROOT / "analysis" / "signals.csv"

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

TEAL = "#2A8C8F"
DEEP_BLUE = "#4B7C92"
SLATE = "#5B5E8D"
ROSE = "#CA4A7A"
GOLD = "#D4A017"
PURPLE = "#6C4F7F"
MAGENTA = "#9B4F8F"
SOFT_TEAL = "#65B2B5"
BG = "#FDFCFB"
TEXT = "#2C3E50"
GRAY_LIGHT = "#8B95A1"
GRID = "#E8E4E0"
BORDER = "#E5E0DB"
CONDITION_COLORS = {"sufficient": TEAL, "insufficient": ROSE}
ANSWER_LETTERS = ["A", "B", "C", "D"]
CHOICE_COLORS = [TEAL, DEEP_BLUE, GOLD, ROSE]

# Jewel-toned plot palette — ordered for maximum adjacent contrast
PLOT_COLORS = [
    TEAL,       # #2A8C8F  teal
    ROSE,       # #CA4A7A  rose/magenta
    GOLD,       # #D4A017  gold
    SLATE,      # #5B5E8D  slate blue
    MAGENTA,    # #9B4F8F  purple-pink
    SOFT_TEAL,  # #65B2B5  light teal
    PURPLE,     # #6C4F7F  deep purple
    DEEP_BLUE,  # #4B7C92  steel blue
    "#B06D94",  #          dusty rose
    "#7F6B9C",  #          lavender
]

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Pre-Action UQ", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #2C3E50; }
h1 { font-family: 'Playfair Display', serif !important; font-weight: 400 !important; font-size: 1.6rem !important; }
h2 { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 1.3rem !important; }
h3 { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 1.1rem !important; }
h4 { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.9rem !important; }
.stApp { background-color: #FDFCFB; }
[data-testid="stMetric"] { background: transparent; border: 1px solid #E5E0DB;
    border-radius: 8px; padding: 10px; }
[data-testid="stMetricLabel"] { font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.04em; color: #8B95A1; }
[data-testid="stMetricValue"] { font-size: 22px; font-weight: 400; }
[data-testid="stSidebar"] { background-color: #F5F3F1; }
footer, #MainMenu, [data-testid="stDeployButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


def _base_layout(**kwargs) -> dict:
    layout = dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        margin=dict(l=50, r=30, t=40, b=40),
        xaxis=dict(gridcolor=GRID, zeroline=False, hoverformat=".2f"),
        yaxis=dict(gridcolor=GRID, zeroline=False, hoverformat=".2f"),
    )
    layout.update(kwargs)
    return layout


def _round_hover(fig: go.Figure) -> go.Figure:
    """Set hovertemplate on all traces to show values rounded to 2dp."""
    for trace in fig.data:
        if trace.hovertemplate is None:
            if hasattr(trace, "orientation") and getattr(trace, "orientation", None) == "h":
                trace.hovertemplate = "%{y}: %{x:.2f}<extra>%{fullData.name}</extra>"
            else:
                trace.hovertemplate = "%{x:.2f}, %{y:.2f}<extra>%{fullData.name}</extra>"
    return fig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_run_prefix(stem: str) -> str:
    parts = stem.split("_")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) if parts else stem


def _fast_file_stats(path: Path) -> dict:
    """Byte-count stats without full JSON parse."""
    try:
        content = path.read_bytes()
    except OSError:
        return {"count": 0, "correct": 0, "incorrect": 0}
    return {
        "count": content.count(b'"question_id"'),
        "correct": content.count(b'"correct": true'),
        "incorrect": content.count(b'"correct": false'),
    }


def _extract_config_head(path: Path) -> dict | None:
    """Read config from the first 4KB of a result JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(4096)
        start = head.find('"config"')
        if start == -1:
            return None
        brace_start = head.find("{", start)
        if brace_start == -1:
            return None
        depth = 0
        for i in range(brace_start, len(head)):
            if head[i] == "{":
                depth += 1
            elif head[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(head[brace_start:i + 1])
    except Exception:
        pass
    return None


def get_result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    all_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    seen: dict[str, Path] = {}
    deduped: list[Path] = []
    for fp in all_files:
        prefix = _extract_run_prefix(fp.stem)
        if prefix not in seen:
            seen[prefix] = fp
            deduped.append(fp)
    return deduped


def _load_json(path: Path) -> dict | None:
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except ImportError:
        pass
    except Exception:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_signals() -> pd.DataFrame | None:
    if SIGNALS_CSV.exists():
        return pd.read_csv(SIGNALS_CSV)
    return None


@st.cache_data(ttl=300)
def load_article_texts() -> dict[str, str]:
    """Load article_id → article_text mapping from the dataset."""
    dataset_path = PROJECT_ROOT / "data" / "quality_all.jsonl"
    if not dataset_path.exists():
        return {}
    articles: dict[str, str] = {}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            articles[obj["article_id"]] = obj["article"]
    return articles


def _fmt_sec(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def format_run_label(cfg: dict) -> str:
    """Human-readable run label."""
    parts = []
    pm = cfg.get("prompt_mode", "direct")
    parts.append("CoT" if pm == "cot" else "direct")
    if cfg.get("think", False):
        parts.append("think")
    else:
        parts.append("nothink")
    parts.append("shuffle" if cfg.get("shuffle_options", True) else "noshuffle")
    parts.append(cfg.get("context_condition", "?"))
    return " · ".join(parts)


def compute_timing(file_path: Path, done_q: int, total_q: int,
                    completed_at: str | None = None) -> dict:
    """Compute elapsed time and ETA.

    Uses the filename timestamp as the start time. For the end time:
      - If completed_at is set (ISO 8601 from final save), use that.
      - If still in progress, use the file's mtime.
      - Never use mtime for finished runs (Dropbox sync keeps updating it).
    """
    elapsed_str = ""
    remaining_str = ""
    pct = done_q / max(total_q, 1)
    finished = pct >= 1.0

    try:
        # Parse start time from filename: ..._YYYYMMDD_HHMMSS.json
        parts = file_path.stem.split("_")
        start = None
        if len(parts) >= 2 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
            try:
                start = datetime.strptime(parts[-2] + "_" + parts[-1], "%Y%m%d_%H%M%S")
            except ValueError:
                pass

        if start is not None:
            if finished and completed_at:
                # Use the recorded completion timestamp
                end = datetime.fromisoformat(completed_at).replace(tzinfo=None)
                elapsed = (end - start).total_seconds()
                elapsed_str = _fmt_sec(elapsed)
                remaining_str = "Done"
            elif finished:
                # No completed_at recorded — just show Done
                remaining_str = "Done"
            else:
                # Still running — use UTC now (filename timestamp is UTC)
                end = datetime.utcnow()
                elapsed = (end - start).total_seconds()
                elapsed_str = _fmt_sec(elapsed)
                if 0 < pct < 1 and elapsed > 0:
                    remaining = elapsed / pct * (1 - pct)
                    remaining_str = f"~{_fmt_sec(remaining)}"
    except Exception:
        pass

    return {"pct": pct, "elapsed": elapsed_str, "remaining": remaining_str}


def results_to_df(all_data: dict[str, dict]) -> pd.DataFrame:
    """Convert loaded result dicts to a flat DataFrame."""
    rows = []
    for run_name, data in all_data.items():
        cfg = data.get("config", {})
        context = cfg.get("context_condition", "unknown")
        think = cfg.get("think", False)
        prompt_mode = cfg.get("prompt_mode", "direct")
        shuffle = cfg.get("shuffle_options", True)

        for qr in data.get("question_results", []):
            if qr.get("skipped", False) or not qr.get("mean_probs"):
                continue

            query_log = qr.get("query_log", [])
            n_queries = len(query_log)
            mean_probs = qr.get("mean_probs", [0.25] * 4)
            final_answer = qr.get("final_answer", 0)

            query_probs = [ql.get("canonical_probs", [0.25]*4) for ql in query_log
                           if len(ql.get("canonical_probs", [])) == 4]
            if query_probs:
                per_query_argmax = [int(np.argmax(p)) for p in query_probs]
                agreement = sum(1 for a in per_query_argmax if a == final_answer) / len(per_query_argmax)
            else:
                agreement = float("nan")

            rows.append({
                "run_name": run_name,
                "context_condition": context,
                "think": think,
                "prompt_mode": prompt_mode,
                "shuffle": shuffle,
                "question_id": qr.get("question_id", ""),
                "article_id": qr.get("article_id", ""),
                "question_text": qr.get("question_text", ""),
                "options": qr.get("options", []),
                "correct_answer": qr.get("correct_answer"),
                "difficult": qr.get("difficult", False),
                "final_answer": final_answer,
                "is_correct": qr.get("correct"),
                "num_queries": n_queries,
                "mean_probs": mean_probs,
                "msp": max(mean_probs) if mean_probs else float("nan"),
                "agreement": agreement,
                "query_log": query_log,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("# Pre-Action UQ")

result_files = get_result_files()
if not result_files:
    st.sidebar.warning("No result files in results/")
    st.title("Pre-Action Uncertainty Quantification")
    st.info("No results found. Run experiments first.")
    st.stop()

# Auto-refresh
auto_refresh = st.sidebar.toggle("Auto-refresh (30s)", value=False)
if auto_refresh:
    st_autorefresh(interval=30_000, key="auto_refresh_counter")

# Build run labels
run_labels: dict[str, Path] = {}
for fp in result_files:
    cfg = _extract_config_head(fp)
    if cfg:
        label = format_run_label(cfg)
    else:
        label = _extract_run_prefix(fp.stem).replace("quality_pilot_", "")
    run_labels[label] = fp

# Select all checkbox + multiselect
select_all = st.sidebar.checkbox("Select all", value=True)
if select_all:
    selected_labels = list(run_labels.keys())
else:
    selected_labels = st.sidebar.multiselect(
        "Experiment runs", list(run_labels.keys()), default=list(run_labels.keys())
    )

selected_paths = {label: run_labels[label] for label in selected_labels}

# Load selected data
all_data: dict[str, dict] = {}
for label, fp in selected_paths.items():
    data = _load_json(fp)
    if data:
        prefix = _extract_run_prefix(fp.stem)
        all_data[prefix] = data

df = results_to_df(all_data)
signals_df = load_signals()


# ---------------------------------------------------------------------------
# Tab 1: Progress
# ---------------------------------------------------------------------------

def tab_progress() -> None:
    st.header("Experiment Progress")

    if not selected_paths:
        st.info("Select at least one run.")
        return

    # One row per run
    for label, fp in selected_paths.items():
        cfg = _extract_config_head(fp)
        stats = _fast_file_stats(fp)
        total_q = (cfg.get("max_questions") if cfg else None) or 4609
        done_q = stats["count"]
        correct = stats["correct"]
        incorrect = stats["incorrect"]
        acc = correct / max(correct + incorrect, 1)
        # Read completed_at from the file head (it's near the top, before question_results)
        completed_at = None
        try:
            with open(fp, encoding="utf-8") as _f:
                head = _f.read(2048)
            import re as _re
            m = _re.search(r'"completed_at":\s*"([^"]+)"', head)
            if m:
                completed_at = m.group(1)
        except Exception:
            pass
        timing = compute_timing(fp, done_q, total_q, completed_at=completed_at)

        col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
        with col1:
            st.markdown(f"**{label}**")
            st.progress(timing["pct"])
        with col2:
            st.metric("Questions", f"{done_q}/{total_q}")
        with col3:
            st.metric("Accuracy", f"{acc:.0%}")
        with col4:
            skipped = done_q - correct - incorrect
            st.metric("Skipped", skipped)
        with col5:
            st.metric("Elapsed", timing["elapsed"] or "-")
        with col6:
            st.metric("ETA", timing["remaining"] or "-")

        st.divider()


# ---------------------------------------------------------------------------
# Tab 2: Uncertainty Distributions
# ---------------------------------------------------------------------------

def tab_distributions() -> None:
    st.header("Uncertainty Distributions")

    if df.empty:
        st.info("No data.")
        return

    active = df[(df["num_queries"] > 0) & df["is_correct"].notna()].copy()
    if active.empty:
        st.info("No valid data.")
        return

    # --- Filters for each experimental variable ---
    filter_cols = st.columns(3)
    with filter_cols[0]:
        ctx_opts = sorted(active["context_condition"].unique())
        selected_ctx = st.selectbox("Context", ["All"] + ctx_opts)
    with filter_cols[1]:
        if "shuffle" in active.columns and active["shuffle"].nunique() > 1:
            shuffle_opts = ["All", "Shuffle", "No shuffle"]
            selected_shuffle = st.selectbox("Shuffle", shuffle_opts)
        else:
            selected_shuffle = "All"
    with filter_cols[2]:
        if "prompt_mode" in active.columns and active["prompt_mode"].nunique() > 1:
            mode_opts = ["All"] + sorted(active["prompt_mode"].unique())
            selected_mode = st.selectbox("Mode", mode_opts)
        else:
            selected_mode = "All"

    if selected_ctx != "All":
        active = active[active["context_condition"] == selected_ctx]
    if selected_shuffle == "Shuffle":
        active = active[active["shuffle"] == True]
    elif selected_shuffle == "No shuffle":
        active = active[active["shuffle"] == False]
    if selected_mode != "All":
        active = active[active["prompt_mode"] == selected_mode]

    if active.empty:
        st.info("No data matches the selected filters.")
        return

    # --- Helper: build a histogram figure ---
    def _msp_hist(sub, label):
        fig = go.Figure()
        correct_msp = sub[sub["is_correct"] == True]["msp"].dropna()
        incorrect_msp = sub[sub["is_correct"] == False]["msp"].dropna()
        if len(correct_msp) > 0:
            fig.add_trace(go.Histogram(x=correct_msp, name="Correct", histnorm="percent",
                                       marker_color=TEAL, opacity=0.7, nbinsx=20))
        if len(incorrect_msp) > 0:
            fig.add_trace(go.Histogram(x=incorrect_msp, name="Incorrect", histnorm="percent",
                                       marker_color=GOLD, opacity=0.7, nbinsx=20))
        fig.update_layout(**_base_layout(title=label, xaxis_title="MSP", yaxis_title="% of group",
                                         barmode="overlay", height=250, showlegend=False))
        return fig

    def _agree_hist(sub, label):
        fig = go.Figure()
        correct_ag = sub[sub["is_correct"] == True]["agreement"].dropna()
        incorrect_ag = sub[sub["is_correct"] == False]["agreement"].dropna()
        if len(correct_ag) > 0:
            fig.add_trace(go.Histogram(x=correct_ag, name="Correct", histnorm="percent",
                                       marker_color=TEAL, opacity=0.7, nbinsx=20))
        if len(incorrect_ag) > 0:
            fig.add_trace(go.Histogram(x=incorrect_ag, name="Incorrect", histnorm="percent",
                                       marker_color=GOLD, opacity=0.7, nbinsx=20))
        fig.update_layout(**_base_layout(title=label, xaxis_title="Agreement", yaxis_title="% of group",
                                         barmode="overlay", height=250, showlegend=False))
        fig.update_xaxes(range=[0, 1.05])
        return fig

    def _run_label(run_name):
        cfg = _extract_config_head(selected_paths.get(
            next((l for l, p in selected_paths.items()
                  if _extract_run_prefix(p.stem) == run_name), ""), None))
        return format_run_label(cfg) if cfg else run_name.replace("quality_", "")

    # --- Per-run MSP distributions (2 per row) ---
    st.subheader("MSP by Run (Correct vs Incorrect)")
    st.caption("Teal = correct, Gold = incorrect")
    run_names = sorted(active["run_name"].unique())
    for i in range(0, len(run_names), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(run_names):
                break
            sub = active[active["run_name"] == run_names[idx]]
            with col:
                st.plotly_chart(_round_hover(_msp_hist(sub, _run_label(run_names[idx]))),
                                use_container_width=True)

    # --- Agreement (only for multi-query runs, 2 per row) ---
    multi = active[(active["num_queries"] > 1) & active["agreement"].notna()]
    if not multi.empty:
        st.subheader("Agreement by Run (Correct vs Incorrect)")
        multi_runs = sorted(multi["run_name"].unique())
        for i in range(0, len(multi_runs), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(multi_runs):
                    break
                sub = multi[multi["run_name"] == multi_runs[idx]]
                with col:
                    st.plotly_chart(_round_hover(_agree_hist(sub, _run_label(multi_runs[idx]))),
                                    use_container_width=True)

    # --- Fragile confidence: MSP by agreement bin ---
    st.subheader("Fragile Confidence: MSP by Agreement Level")
    st.caption(
        "Box plots of MSP at each agreement level. "
        "High MSP + low agreement = fragile confidence (overconfident on unstable answers)."
    )
    multi_fc = active[(active["num_queries"] > 1) & active["agreement"].notna()]
    if not multi_fc.empty:
        def _fragile_box(sub, label):
            sub = sub.copy()
            sub["agree_bin"] = (sub["agreement"] * 10).round() / 10
            fig3 = go.Figure()
            for correct_val, color, name in [(True, TEAL, "Correct"), (False, GOLD, "Incorrect")]:
                s = sub[sub["is_correct"] == correct_val]
                if len(s) > 0:
                    fig3.add_trace(go.Box(x=s["agree_bin"], y=s["msp"], name=name,
                                          marker_color=color, boxpoints="outliers", line_width=1.5))
            fig3.update_layout(**_base_layout(title=label, xaxis_title="Agreement", yaxis_title="MSP",
                                              boxmode="group", height=280, showlegend=False))
            fig3.update_xaxes(dtick=0.1)
            return fig3

        fc_runs = [rn for rn in sorted(multi_fc["run_name"].unique())
                   if len(multi_fc[multi_fc["run_name"] == rn]) >= 5]
        for i in range(0, len(fc_runs), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(fc_runs):
                    break
                sub = multi_fc[multi_fc["run_name"] == fc_runs[idx]].dropna(subset=["agreement", "msp"])
                with col:
                    st.plotly_chart(_round_hover(_fragile_box(sub, _run_label(fc_runs[idx]))),
                                    use_container_width=True)
    else:
        st.caption("Requires multi-query runs (shuffle conditions).")


# ---------------------------------------------------------------------------
# Tab 3: Condition Comparison
# ---------------------------------------------------------------------------

def tab_comparison() -> None:
    st.header("Condition Comparison")

    if df.empty:
        st.info("No data.")
        return

    # Accuracy table — dynamically split by all factors that vary across runs
    st.subheader("Accuracy by Condition")
    active = df[df["num_queries"] > 0].copy()
    if active.empty:
        st.info("No valid data.")
        return

    # Detect which factors actually vary (only show columns that differ)
    factor_cols = []
    factor_labels = {
        "prompt_mode": "Mode",
        "think": "Think",
        "shuffle": "Shuffle",
        "context_condition": "Context",
    }
    for col, label in factor_labels.items():
        if col in active.columns and active[col].nunique() > 1:
            factor_cols.append(col)

    # If no factors vary, just show one row
    if not factor_cols:
        factor_cols = ["context_condition"]

    # Group by the varying factors
    pivot_data = []
    for keys, sub in active.groupby(factor_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(sub) == 0:
            continue
        row: dict = {}
        for col, val in zip(factor_cols, keys):
            label = factor_labels.get(col, col)
            if isinstance(val, bool):
                row[label] = "Yes" if val else "No"
            else:
                row[label] = str(val).capitalize()
        acc = sub["is_correct"].mean()
        easy = sub[~sub["difficult"]]
        hard = sub[sub["difficult"]]
        acc_easy = easy["is_correct"].mean() if len(easy) > 0 else float("nan")
        acc_hard = hard["is_correct"].mean() if len(hard) > 0 else float("nan")

        msp_all = sub["msp"].mean()
        msp_easy = easy["msp"].mean() if len(easy) > 0 else float("nan")
        msp_hard = hard["msp"].mean() if len(hard) > 0 else float("nan")

        row["Accuracy"] = f"{acc:.0%}"
        row["Acc Easy"] = f"{acc_easy:.0%}" if not math.isnan(acc_easy) else "-"
        row["Acc Hard"] = f"{acc_hard:.0%}" if not math.isnan(acc_hard) else "-"
        row["Confidence"] = f"{msp_all:.2f}"
        row["Conf Easy"] = f"{msp_easy:.2f}" if not math.isnan(msp_easy) else "-"
        row["Conf Hard"] = f"{msp_hard:.2f}" if not math.isnan(msp_hard) else "-"
        row["N"] = len(sub)
        pivot_data.append(row)

    if pivot_data:
        # Build HTML table with grouped column headers
        factor_hdrs = [factor_labels.get(c, c) for c in factor_cols]
        n_factors = len(factor_hdrs)

        html = """<style>
        .cond-table { border-collapse: collapse; width: 100%; font-family: Inter, sans-serif; font-size: 13px; }
        .cond-table th, .cond-table td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #E5E0DB; }
        .cond-table th { color: #8B95A1; font-weight: 500; }
        .cond-table th.group { border-bottom: 2px solid #2A8C8F; color: #2C3E50; font-weight: 600; }
        .cond-table td.factor { text-align: left; font-weight: 500; }
        .cond-table tr:hover { background: #F5F3F1; }
        </style><table class="cond-table">"""

        # Row 1: group headers
        html += "<tr>"
        for h in factor_hdrs:
            html += f'<th rowspan="2" class="factor">{h}</th>'
        html += '<th colspan="2" class="group">Overall</th>'
        html += '<th colspan="2" class="group">Easy</th>'
        html += '<th colspan="2" class="group">Hard</th>'
        html += '<th rowspan="2">N</th></tr>'

        # Row 2: sub-headers
        html += "<tr>"
        for _ in range(3):
            html += "<th>Acc</th><th>Conf</th>"
        html += "</tr>"

        # Data rows
        for row in pivot_data:
            html += "<tr>"
            for h in factor_hdrs:
                html += f'<td class="factor">{row.get(h, "")}</td>'
            html += f'<td>{row["Accuracy"]}</td><td>{row["Confidence"]}</td>'
            html += f'<td>{row["Acc Easy"]}</td><td>{row["Conf Easy"]}</td>'
            html += f'<td>{row["Acc Hard"]}</td><td>{row["Conf Hard"]}</td>'
            html += f'<td>{row["N"]}</td>'
            html += "</tr>"

        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

    # Signals table from CSV
    if signals_df is not None and not signals_df.empty:
        st.subheader("Key Signals by Condition")
        signal_cols = ["msp", "single_entropy", "agreement", "epistemic",
                       "mean_confidence", "answer_coverage", "hesitation_mass",
                       "mean_pairwise_jsd", "confidence_variance"]
        available = [c for c in signal_cols if c in signals_df.columns]

        if available:
            summary_rows = []
            for cond in sorted(signals_df["condition"].unique()):
                sub = signals_df[signals_df["condition"] == cond]
                row = {"Condition": cond.replace("quality_pilot_", "")}
                for col in available:
                    vals = sub[col].dropna()
                    row[col] = f"{vals.mean():.3f}" if len(vals) > 0 else "-"
                summary_rows.append(row)
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Calibration curves
    st.subheader("Calibration: Reliability Diagram")
    st.caption(
        "Top: binned reliability (mean confidence vs observed accuracy). "
        "Bottom: bin counts. ECE = weighted mean |accuracy - confidence|."
    )
    _plot_calibration_reliability(df)


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 1.0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _plot_calibration_reliability(df: pd.DataFrame, n_bins: int = 16, min_count: int = 2) -> None:
    """Binned reliability diagram — clean lines, no bin count panel."""
    df_valid = df[(df["num_queries"] > 0) & df["is_correct"].notna()].copy()
    if df_valid.empty:
        st.info("No data for calibration.")
        return

    fig = go.Figure()

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0.25, 1], y=[0.25, 1], mode="lines",
        line=dict(color=GRAY_LIGHT, dash="dash", width=1),
        name="Perfect calibration", showlegend=True,
    ))

    run_names = sorted(df_valid["run_name"].unique())
    for ri, run_name in enumerate(run_names):
        sub = df_valid[df_valid["run_name"] == run_name]
        if len(sub) < 5:
            continue

        color = PLOT_COLORS[ri % len(PLOT_COLORS)]

        # Build clean label from config
        cfg = _extract_config_head(selected_paths.get(
            next((l for l, p in selected_paths.items()
                  if _extract_run_prefix(p.stem) == run_name), ""), None))
        label = format_run_label(cfg) if cfg else run_name.replace("quality_", "")

        msp_vals = sub["msp"].values
        correct_vals = sub["is_correct"].astype(float).values

        # Fixed bins across 0.25–1.0
        lo, hi = 0.25, 1.0
        bin_edges = np.linspace(lo, hi, n_bins + 1)

        bin_centers = []
        bin_accs = []
        bin_counts = []

        for b in range(n_bins):
            if b < n_bins - 1:
                mask = (msp_vals >= bin_edges[b]) & (msp_vals < bin_edges[b + 1])
            else:
                mask = (msp_vals >= bin_edges[b]) & (msp_vals <= bin_edges[b + 1])
            n_in_bin = mask.sum()
            if n_in_bin < min_count:
                bin_accs.append(float("nan"))
                bin_centers.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
                bin_counts.append(int(n_in_bin))
                continue

            bin_centers.append(float(msp_vals[mask].mean()))
            bin_accs.append(float(correct_vals[mask].mean()))
            bin_counts.append(int(n_in_bin))

        # ECE
        ece = 0.0
        total_n = sum(bin_counts)
        for acc, conf, n in zip(bin_accs, bin_centers, bin_counts):
            if not math.isnan(acc) and total_n > 0:
                ece += (n / total_n) * abs(acc - conf)

        # Plot valid bins as a clean line
        valid = [(c, a) for c, a in zip(bin_centers, bin_accs) if not math.isnan(a)]
        if valid:
            plot_x, plot_y = zip(*valid)
            fig.add_trace(go.Scatter(
                x=list(plot_x), y=list(plot_y),
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{label} (ECE={ece:.3f})",
            ))

    fig.update_layout(**_base_layout(
        title="Calibration: Confidence vs Accuracy",
        xaxis_title="MSP (model confidence)",
        yaxis_title="Accuracy in bin",
        xaxis=dict(range=[0.2, 1.02], gridcolor=GRID),
        yaxis=dict(range=[0, 1.05], gridcolor=GRID),
        legend=dict(x=0.02, y=0.98),
        height=450,
    ))

    st.plotly_chart(_round_hover(fig), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Question Explorer
# ---------------------------------------------------------------------------

def tab_explorer() -> None:
    st.header("Question Explorer")

    if df.empty:
        st.info("No data.")
        return

    df_valid = df[df["num_queries"] > 0].copy()
    if df_valid.empty:
        st.info("No questions with valid queries.")
        return

    df_valid["label"] = (
        df_valid["question_id"].str[:20] + " | " +
        df_valid["question_text"].str[:50] + " | " +
        df_valid["context_condition"]
    )

    selected = st.selectbox("Select a question", df_valid["label"].unique())
    if not selected:
        return

    row = df_valid[df_valid["label"] == selected].iloc[0]

    # Question text and options (compact)
    st.markdown(f"**Q: {row['question_text']}**", help=None)
    options = row.get("options", [])
    correct = row.get("correct_answer", -1)
    opts_html = ""
    for i, opt in enumerate(options):
        marker = ' <span style="color:#2A8C8F; font-weight:600;">[CORRECT]</span>' if i == correct else ""
        opts_html += f'<div style="font-size:13px; margin:2px 0;"><b>{ANSWER_LETTERS[i]})</b> {opt}{marker}</div>'
    st.markdown(opts_html, unsafe_allow_html=True)

    # Compact metadata line instead of large metric cards
    diff_label = "Hard" if row["difficult"] else "Easy"
    correct_label = "Yes" if row["is_correct"] else "No"
    agree_str = f'{row["agreement"]:.2f}' if not math.isnan(row["agreement"]) else "-"
    st.markdown(
        f"**Difficulty:** {diff_label} · "
        f"**Context:** {row['context_condition'].capitalize()} · "
        f"**Answer:** {ANSWER_LETTERS[row['final_answer']]} · "
        f"**Correct:** {correct_label} · "
        f"**MSP:** {row['msp']:.3f} · "
        f"**Agreement:** {agree_str}"
    )

    # Show article context
    article_texts = load_article_texts()
    article_id = row.get("article_id", "")
    article_text = article_texts.get(article_id, "")
    if article_text:
        with st.expander(f"Article context (article_id: {article_id})", expanded=False):
            truncated = article_text
            # Use a styled div — st.text uses <pre> which doesn't wrap
            escaped = truncated.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Collapse Gutenberg hard-wrapped lines: single \n → space, \n\n → paragraph break
            paragraphs = escaped.split("\n\n")
            html = "".join(f"<p>{' '.join(p.split())}</p>" for p in paragraphs)
            st.markdown(
                f'<div style="font-size:14px; line-height:1.6; max-height:500px; '
                f'overflow-y:auto; white-space:normal; word-wrap:break-word;">'
                f'{html}</div>',
                unsafe_allow_html=True,
            )

    # Per-query bar chart
    query_log = row.get("query_log", [])
    if query_log:
        st.subheader("Per-Query Probability Distributions")
        fig = go.Figure()
        for qi, ql in enumerate(query_log):
            probs = ql.get("canonical_probs", [0.25]*4)
            for li, letter in enumerate(ANSWER_LETTERS):
                fig.add_trace(go.Bar(
                    x=[f"Q{qi}"], y=[probs[li] if li < len(probs) else 0],
                    name=letter if qi == 0 else None,
                    marker_color=CHOICE_COLORS[li],
                    showlegend=(qi == 0), legendgroup=letter,
                ))
        fig.update_layout(**_base_layout(
            title="A/B/C/D Probabilities per Query",
            xaxis_title="Query", yaxis_title="Probability", barmode="stack",
        ))
        st.plotly_chart(_round_hover(fig), use_container_width=True)

    # Same question across conditions
    same_q = df_valid[df_valid["question_id"] == row["question_id"]]
    if len(same_q) > 1:
        st.subheader("Same Question Across Conditions")
        for _, other in same_q.iterrows():
            ctx = other["context_condition"]
            ans = ANSWER_LETTERS[other["final_answer"]]
            correct_str = "Correct" if other["is_correct"] else "Wrong"
            st.write(f"**{ctx.capitalize()}**: Answer={ans}, MSP={other['msp']:.3f}, {correct_str}")


# ---------------------------------------------------------------------------
# Tab 5: Effect Analysis
# ---------------------------------------------------------------------------

def tab_effects() -> None:
    """Matched-pair analysis: what does each experimental variable do?"""
    st.header("Effect Analysis")
    st.caption("Each delta is the average across matched pairs differing only on that variable.")

    if df.empty or len(df["run_name"].unique()) < 2:
        st.info("Need at least 2 runs to compare effects.")
        return

    active = df[df["num_queries"] > 0].copy()

    # Index runs by condition tuple: (prompt_mode, shuffle, context)
    run_metrics: dict[str, dict] = {}
    for run_name in active["run_name"].unique():
        sub = active[active["run_name"] == run_name]
        valid = sub[sub["is_correct"].notna()]
        multi = sub[sub["agreement"].notna()]
        correct_sub = valid[valid["is_correct"] == True]
        incorrect_sub = valid[valid["is_correct"] == False]

        run_metrics[run_name] = {
            "accuracy": valid["is_correct"].mean() if len(valid) > 0 else None,
            "n": len(valid),
            "msp_correct": correct_sub["msp"].mean() if len(correct_sub) > 0 else None,
            "msp_incorrect": incorrect_sub["msp"].mean() if len(incorrect_sub) > 0 else None,
            "agreement_correct": correct_sub["agreement"].mean() if len(correct_sub[correct_sub["agreement"].notna()]) > 0 else None,
            "agreement_incorrect": incorrect_sub["agreement"].mean() if len(incorrect_sub[incorrect_sub["agreement"].notna()]) > 0 else None,
            "prompt_mode": sub["prompt_mode"].iloc[0] if "prompt_mode" in sub.columns else "direct",
            "shuffle": sub["shuffle"].iloc[0] if "shuffle" in sub.columns else True,
            "context": sub["context_condition"].iloc[0],
        }

    # --- Summary metrics table (like old dashboard) ---
    st.subheader("Summary Metrics by Run")
    summary_rows = []
    for run_name, m in sorted(run_metrics.items()):
        # Build label from config
        cfg = _extract_config_head(selected_paths.get(
            next((l for l, p in selected_paths.items()
                  if _extract_run_prefix(p.stem) == run_name), ""), None))
        label = format_run_label(cfg) if cfg else run_name.replace("quality_", "")

        summary_rows.append({
            "Run": label,
            "N": m["n"],
            "Accuracy": f"{m['accuracy']:.1%}" if m["accuracy"] is not None else "-",
            "MSP (correct)": f"{m['msp_correct']:.3f}" if m["msp_correct"] is not None else "-",
            "MSP (incorrect)": f"{m['msp_incorrect']:.3f}" if m["msp_incorrect"] is not None else "-",
            "Agreement (correct)": f"{m['agreement_correct']:.3f}" if m["agreement_correct"] is not None else "-",
            "Agreement (incorrect)": f"{m['agreement_incorrect']:.3f}" if m["agreement_incorrect"] is not None else "-",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # --- Matched pair analysis ---
    st.subheader("Matched Pair Effects")

    # Detect which variables vary
    contexts = set(m["context"] for m in run_metrics.values())
    shuffles = set(m["shuffle"] for m in run_metrics.values())
    modes = set(m["prompt_mode"] for m in run_metrics.values())

    # Find matched pairs for each variable that has both levels
    variables = []
    if len(contexts) > 1:
        variables.append(("context", "Context: sufficient vs insufficient"))
    if len(shuffles) > 1:
        variables.append(("shuffle", "Shuffle: on vs off"))
    if len(modes) > 1:
        variables.append(("prompt_mode", "Mode: direct vs CoT"))

    if not variables:
        st.info("Need runs with contrasting conditions to compute effects.")
        return

    effect_rows = []
    for var_name, var_label in variables:
        # Group runs into pairs differing only on this variable
        pairs_found = 0
        acc_deltas = []
        msp_deltas = []

        run_list = list(run_metrics.items())
        for i, (name_a, m_a) in enumerate(run_list):
            for name_b, m_b in run_list[i+1:]:
                # Check they differ on var_name and match on everything else
                diff_on_var = m_a[var_name] != m_b[var_name]
                other_vars = [v for v in ["context", "shuffle", "prompt_mode"] if v != var_name]
                match_on_rest = all(m_a[v] == m_b[v] for v in other_vars)

                if diff_on_var and match_on_rest:
                    pairs_found += 1
                    if m_a["accuracy"] is not None and m_b["accuracy"] is not None:
                        acc_deltas.append(m_a["accuracy"] - m_b["accuracy"])

        avg_delta = sum(acc_deltas) / len(acc_deltas) if acc_deltas else None
        effect_rows.append({
            "Effect": var_label,
            "Pairs": pairs_found,
            "Accuracy Delta": f"{avg_delta:+.1%}" if avg_delta is not None else "-",
        })

    if effect_rows:
        st.dataframe(pd.DataFrame(effect_rows), use_container_width=True, hide_index=True)

    # --- Per-run distribution comparison ---
    st.subheader("MSP Distributions by Run")
    st.caption("Compare how confidence distributions shift across conditions.")

    fig = go.Figure()
    for ri, run_name in enumerate(sorted(active["run_name"].unique())):
        sub = active[(active["run_name"] == run_name) & active["msp"].notna()]
        if len(sub) == 0:
            continue

        cfg = _extract_config_head(selected_paths.get(
            next((l for l, p in selected_paths.items()
                  if _extract_run_prefix(p.stem) == run_name), ""), None))
        label = format_run_label(cfg) if cfg else run_name.replace("quality_", "")
        color = PLOT_COLORS[ri % len(PLOT_COLORS)]

        fig.add_trace(go.Violin(
            x=sub["msp"], name=label,
            line_color=color, fillcolor=color,
            opacity=0.5, orientation="h",
            side="positive", meanline_visible=True,
        ))

    fig.update_layout(**_base_layout(
        title="MSP Distribution per Run",
        xaxis_title="MSP", yaxis_title="",
        height=max(250, 80 * len(active["run_name"].unique())),
    ))
    st.plotly_chart(_round_hover(fig), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Pre-Action Uncertainty Quantification")
st.caption("QuALITY dataset | Context sufficiency experiments")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Progress", "Uncertainty Distributions",
    "Condition Comparison", "Effect Analysis", "Question Explorer",
])

with tab1:
    tab_progress()
with tab2:
    tab_distributions()
with tab3:
    tab_comparison()
with tab4:
    tab_effects()
with tab5:
    tab_explorer()
