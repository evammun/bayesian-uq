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

TEAL = "#2D7F83"
DEEP_BLUE = "#1E4D8A"
DARK_TEAL = "#2F555A"
CHARCOAL = "#1A2F32"
SLATE = "#6B7280"
ROSE = "#B85C5C"
GOLD = "#EEB127"
BG = "#FDFCFB"
TEXT = "#2C3E50"
GRAY_LIGHT = "#8B95A1"
GRID = "#E8E4E0"
BORDER = "#E5E0DB"
CONDITION_COLORS = {"sufficient": TEAL, "insufficient": ROSE}
ANSWER_LETTERS = ["A", "B", "C", "D"]
CHOICE_COLORS = [TEAL, DEEP_BLUE, GOLD, ROSE]

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Pre-Action UQ", layout="wide")

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

    CORRECT_COLOR = TEAL
    INCORRECT_COLOR = GOLD

    # Context condition selector
    ctx_options = sorted(active["context_condition"].unique())
    selected_ctx = st.selectbox("Context condition", ["All"] + ctx_options)
    if selected_ctx != "All":
        active = active[active["context_condition"] == selected_ctx]

    # MSP distribution — correct vs incorrect overlay
    st.subheader("Max Single Probability (MSP)")
    fig = go.Figure()
    correct_msp = active[active["is_correct"] == True]["msp"].dropna()
    incorrect_msp = active[active["is_correct"] == False]["msp"].dropna()
    if len(correct_msp) > 0:
        fig.add_trace(go.Histogram(x=correct_msp, name="Correct", histnorm="percent",
                                   marker_color=CORRECT_COLOR, opacity=0.7, nbinsx=20))
    if len(incorrect_msp) > 0:
        fig.add_trace(go.Histogram(x=incorrect_msp, name="Incorrect", histnorm="percent",
                                   marker_color=INCORRECT_COLOR, opacity=0.7, nbinsx=20))
    fig.update_layout(**_base_layout(title="MSP: Correct vs Incorrect",
                                     xaxis_title="MSP", yaxis_title="% of group", barmode="overlay"))
    st.plotly_chart(_round_hover(fig), use_container_width=True)

    # Agreement distribution — correct vs incorrect
    st.subheader("Agreement Rate")
    fig2 = go.Figure()
    correct_ag = active[active["is_correct"] == True]["agreement"].dropna()
    incorrect_ag = active[active["is_correct"] == False]["agreement"].dropna()
    if len(correct_ag) > 0:
        fig2.add_trace(go.Histogram(x=correct_ag, name="Correct", histnorm="percent",
                                    marker_color=CORRECT_COLOR, opacity=0.7, nbinsx=20))
    if len(incorrect_ag) > 0:
        fig2.add_trace(go.Histogram(x=incorrect_ag, name="Incorrect", histnorm="percent",
                                    marker_color=INCORRECT_COLOR, opacity=0.7, nbinsx=20))
    fig2.update_layout(**_base_layout(title="Agreement: Correct vs Incorrect",
                                      xaxis_title="Agreement", yaxis_title="% of group", barmode="overlay"))
    st.plotly_chart(_round_hover(fig2), use_container_width=True)

    # Fragile confidence scatter: agreement (x) vs MSP (y)
    st.subheader("Fragile Confidence: Agreement vs MSP")
    fig3 = go.Figure()
    for cond, color in CONDITION_COLORS.items():
        sub = active[active["context_condition"] == cond].dropna(subset=["agreement", "msp"])
        for correct_val, symbol, label_suffix in [
            (True, "circle", "Correct"),
            (False, "x", "Wrong"),
        ]:
            s = sub[sub["is_correct"] == correct_val]
            if len(s) > 0:
                fig3.add_trace(go.Scatter(
                    x=s["agreement"], y=s["msp"], mode="markers",
                    marker=dict(color=color, symbol=symbol, size=8, opacity=0.7),
                    name=f"{cond.capitalize()} {label_suffix}",
                ))
    fig3.update_layout(**_base_layout(title="Fragile Confidence Space",
                                      xaxis_title="Agreement", yaxis_title="MSP"))
    st.plotly_chart(_round_hover(fig3), use_container_width=True)


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
        acc_easy = sub[~sub["difficult"]]["is_correct"].mean() if (~sub["difficult"]).any() else float("nan")
        acc_hard = sub[sub["difficult"]]["is_correct"].mean() if sub["difficult"].any() else float("nan")
        row["Accuracy"] = f"{acc:.1%}"
        row["Easy"] = f"{acc_easy:.1%}" if not math.isnan(acc_easy) else "-"
        row["Hard"] = f"{acc_hard:.1%}" if not math.isnan(acc_hard) else "-"
        row["N"] = len(sub)
        pivot_data.append(row)

    if pivot_data:
        st.dataframe(pd.DataFrame(pivot_data), use_container_width=True, hide_index=True)

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
    """Binned reliability diagram + sharpness histogram (shared x-axis)."""
    from plotly.subplots import make_subplots

    df_valid = df[(df["num_queries"] > 0) & df["is_correct"].notna()].copy()
    if df_valid.empty:
        st.info("No data for calibration.")
        return

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=["Reliability", "Bin Counts"],
    )

    # Perfect calibration diagonal (top panel)
    fig.add_trace(go.Scatter(
        x=[0.25, 1], y=[0.25, 1], mode="lines",
        line=dict(color=GRAY_LIGHT, dash="dash", width=1),
        name="Perfect", showlegend=True,
    ), row=1, col=1)

    # Distinct colors for each run (cycle through a palette)
    run_palette = [TEAL, ROSE, DEEP_BLUE, GOLD, DARK_TEAL, SLATE, CHARCOAL, "#3A8A8F"]

    run_names = sorted(df_valid["run_name"].unique())
    for ri, run_name in enumerate(run_names):
        sub = df_valid[df_valid["run_name"] == run_name]
        if len(sub) < 5:
            continue

        color = run_palette[ri % len(run_palette)]

        # Build clean label from config
        cfg = _extract_config_head(selected_paths.get(
            next((l for l, p in selected_paths.items()
                  if _extract_run_prefix(p.stem) == run_name), ""), None))
        label = format_run_label(cfg) if cfg else run_name.replace("quality_", "")

        msp_vals = sub["msp"].values
        correct_vals = sub["is_correct"].astype(float).values

        # Fixed bins across 0.25–1.0 range
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
                bin_centers.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
                bin_accs.append(float("nan"))
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

        # Filter NaN bins
        valid = [(c, a, n) for c, a, n in zip(bin_centers, bin_accs, bin_counts) if not math.isnan(a)]
        if valid:
            plot_x, plot_y, _ = zip(*valid)
            # Top panel: clean line only
            fig.add_trace(go.Scatter(
                x=list(plot_x), y=list(plot_y),
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{label} (ECE={ece:.3f})",
            ), row=1, col=1)

        # Bottom panel: bin counts
        all_centers = [(bin_edges[b] + bin_edges[b + 1]) / 2 for b in range(n_bins)]
        fig.add_trace(go.Bar(
            x=all_centers, y=bin_counts,
            marker_color=color, opacity=0.5,
            name=label, showlegend=False,
            width=(hi - lo) / n_bins * 0.8,
        ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        margin=dict(l=50, r=30, t=40, b=40),
        height=550,
        legend=dict(x=0.02, y=0.98),
    )
    fig.update_xaxes(gridcolor=GRID, range=[0.2, 1.02], row=1, col=1)
    fig.update_xaxes(gridcolor=GRID, range=[0.2, 1.02], title_text="MSP (model confidence)", row=2, col=1)
    fig.update_yaxes(gridcolor=GRID, range=[0, 1.05], title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(gridcolor=GRID, title_text="Count", row=2, col=1)

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

    # Question text and options
    st.subheader(f"Q: {row['question_text']}")
    options = row.get("options", [])
    correct = row.get("correct_answer", -1)
    for i, opt in enumerate(options):
        marker = " **[CORRECT]**" if i == correct else ""
        st.write(f"**{ANSWER_LETTERS[i]})** {opt}{marker}")

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
# Main
# ---------------------------------------------------------------------------

st.title("Pre-Action Uncertainty Quantification")
st.caption("QuALITY dataset | Context sufficiency experiments")

tab1, tab2, tab3, tab4 = st.tabs([
    "Progress", "Uncertainty Distributions",
    "Condition Comparison", "Question Explorer",
])

with tab1:
    tab_progress()
with tab2:
    tab_distributions()
with tab3:
    tab_comparison()
with tab4:
    tab_explorer()
