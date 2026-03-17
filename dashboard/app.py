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
from datetime import datetime, timezone
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

def _extract_run_name_prefix(stem: str) -> str:
    """Extract the condition prefix from a result filename, stripping the timestamp.

    Example: 'exp1_full_direct_shuffle_para_20260317_093909' -> 'exp1_full_direct_shuffle_para'
    This groups original and resumed runs for the same condition together.
    """
    # Timestamps are always _YYYYMMDD_HHMMSS at the end
    parts = stem.split("_")
    # Walk backwards: strip trailing parts that look like timestamp components
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) if parts else stem


def get_result_files() -> list[Path]:
    """Scan results/ for JSON files, keeping only the newest per condition.

    When a run is resumed with --resume, both the old partial file and the
    new resumed file exist with different timestamps but the same condition
    prefix. The resumed file supersedes the partial, so we keep only the
    most recent file per condition prefix (by mtime).
    """
    if not RESULTS_DIR.exists():
        return []
    all_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    # Keep only the newest file per condition prefix
    seen_prefixes: dict[str, Path] = {}
    deduped: list[Path] = []
    for fp in all_files:
        prefix = _extract_run_name_prefix(fp.stem)
        if prefix not in seen_prefixes:
            seen_prefixes[prefix] = fp
            deduped.append(fp)
    return deduped


try:
    import orjson as _json_fast
    def _try_load_json(path: str) -> dict | None:
        """Try to load a JSON file once using orjson for speed."""
        try:
            with open(path, "rb") as f:
                return _json_fast.loads(f.read())
        except (_json_fast.JSONDecodeError, OSError):
            return None
except ImportError:
    def _try_load_json(path: str) -> dict | None:  # type: ignore[misc]
        """Try to load a JSON file once. Returns None on any error."""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None


def _strip_unused_fields(data: dict) -> dict:
    """Remove fields the dashboard never reads to reduce memory and parse overhead.

    The dashboard only uses these per-query fields:
      canonical_probs, canonical_answer, query_text
    Everything else is for the analysis pipeline or audit trail — safe to drop.
    For an 11-query × 5,000-question run this cuts in-memory size from ~280MB
    to roughly ~15MB.
    """
    _KEEP_QUERY_KEYS = {"canonical_probs", "canonical_answer", "query_text"}
    for qr in data.get("question_results", []):
        qr.pop("answer_counts", None)
        for ql in qr.get("query_log", []):
            drop_keys = [k for k in ql if k not in _KEEP_QUERY_KEYS]
            for k in drop_keys:
                del ql[k]
    return data


# Session-state cache TTL for result files (seconds). Should be >= the
# auto-refresh interval (60s) so that user interactions between refreshes
# get instant cache hits, and each auto-refresh triggers at most one reload.
_RESULT_CACHE_TTL = 60


def _fast_file_stats(path: str) -> dict | None:
    """Extract question count, accuracy, and avg queries from a result file
    without JSON parsing — pure byte-counting on the raw file.

    Returns dict with keys: count, correct, incorrect, avg_queries (approx).
    Returns None if the file can't be read.
    """
    try:
        with open(path, "rb") as f:
            content = f.read()
    except OSError:
        return None

    count = content.count(b'"question_id"')
    correct = content.count(b'"correct": true')
    incorrect = content.count(b'"correct": false')
    # num_queries appears once per question result as "num_queries": N
    # Count total query_log entries via "query_number" (one per query)
    total_queries = content.count(b'"query_number"')
    avg_queries = total_queries / count if count else 0

    return {
        "count": count,
        "correct": correct,
        "incorrect": incorrect,
        "total_queries": total_queries,
        "avg_queries": avg_queries,
    }


def _count_results_fast(path: str) -> int | None:
    """Count question_results in a JSON file without parsing the whole thing."""
    stats = _fast_file_stats(path)
    return stats["count"] if stats else None


def load_result_file(path: str) -> dict | None:
    """Load a result JSON with stale-while-revalidate caching.

    Strategy:
      1. If we have a cached copy younger than _RESULT_CACHE_TTL, return it.
      2. On TTL expiry, check the file's mtime. If unchanged since last load,
         bump the cache timestamp and return the old data (no re-parse).
      3. If mtime changed, do a fast byte-count to get the new question count.
         If the count hasn't changed (file was rewritten but same data), skip
         the re-parse. If it has changed, only re-parse if enough new results
         have accumulated (_REPARSE_THRESHOLD) — otherwise patch the cached
         count and serve stale data. This avoids re-parsing 300MB files on
         every auto-refresh while a run is in progress.
      4. If the fresh load fails (file mid-write / locked), return the last
         known good copy — the run never disappears from the dashboard.

    Uses st.session_state instead of @st.cache_data because:
      - We need stale-while-revalidate (serve old data on failure)
      - No pickle serialisation overhead for large dicts
      - Full control over TTL and invalidation
    """
    data_key = f"_rc_data_{path}"
    ts_key = f"_rc_ts_{path}"
    mtime_key = f"_rc_mtime_{path}"
    count_key = f"_rc_count_{path}"

    now = time.time()
    cached_data = st.session_state.get(data_key)
    cached_ts = st.session_state.get(ts_key, 0)
    cached_mtime = st.session_state.get(mtime_key, 0)
    cached_count = st.session_state.get(count_key, 0)

    # Cache hit — return without touching the filesystem
    if cached_data is not None and (now - cached_ts) < _RESULT_CACHE_TTL:
        return cached_data

    # TTL expired — check if the file has actually changed before re-parsing
    try:
        current_mtime = Path(path).stat().st_mtime
    except OSError:
        return cached_data  # file inaccessible, serve stale

    if cached_data is not None and current_mtime == cached_mtime:
        # File unchanged — bump cache timestamp, skip the expensive re-parse
        st.session_state[ts_key] = now
        return cached_data

    # File changed — do a fast byte-count to see how many results exist now.
    # Only trigger a full re-parse if enough new results have accumulated
    # (500 questions ≈ 10% of a full run). This prevents re-parsing 300MB+
    # files every 30s while a run is in progress.
    _REPARSE_THRESHOLD = 500
    current_count = _count_results_fast(path)
    if (
        cached_data is not None
        and current_count is not None
        and current_count - cached_count < _REPARSE_THRESHOLD
    ):
        # Not enough new data to justify a full re-parse — just update the
        # count so the Progress tab shows updated numbers, and serve stale
        st.session_state[ts_key] = now
        st.session_state[mtime_key] = current_mtime
        st.session_state[count_key] = current_count
        # Patch the cached data's question count for the Progress tab
        cached_data["_fast_count"] = current_count
        return cached_data

    # Enough new results (or first load) — do a full re-parse
    fresh = _try_load_json(path)
    if fresh is not None:
        fresh = _strip_unused_fields(fresh)
        fresh_count = len(fresh.get("question_results", []))
        fresh["_fast_count"] = fresh_count
        st.session_state[data_key] = fresh
        st.session_state[ts_key] = now
        st.session_state[mtime_key] = current_mtime
        st.session_state[count_key] = fresh_count
        return fresh

    # Fresh load failed — serve stale data if we have it (run stays visible)
    if cached_data is not None:
        return cached_data

    # First load and file is unreadable — nothing we can do
    return None


def _extract_config_from_file(path: str) -> dict | None:
    """Read only the config from a result JSON without loading full question_results.

    Result files can be 100MB+. This reads just enough to extract the config
    object (always in the first 4KB), avoiding the cost of deserialising
    thousands of question results. Single attempt, no blocking retries —
    if this fails, the caller falls back to _config_from_filename().
    """
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(4096)
        start = head.find('"config"')
        if start == -1:
            return None
        brace_start = head.find("{", start)
        if brace_start == -1:
            return None
        # Walk forward counting braces to find the matching close
        depth = 0
        for i in range(brace_start, len(head)):
            if head[i] == "{":
                depth += 1
            elif head[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(head[brace_start:i + 1])
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _config_from_filename(stem: str) -> dict | None:
    """Infer a minimal config dict from the result filename.

    Filenames follow the pattern: exp1_{scale}_{prompt}_{shuffle}_{para}_{timestamp}
    e.g. "exp1_full_direct_shuffle_nopara_20260316_134902"

    This is a last-resort fallback when the file can't be read (mid-write).
    The resulting dict has enough fields for format_run_name() to work.
    """
    parts = stem.lower().split("_")
    cfg: dict = {"model": "qwen3:8b-q4_K_M"}  # default model

    # Prompt mode
    if "direct" in parts:
        cfg["prompt_mode"] = "direct"
    elif "cotstruct" in parts:
        cfg["prompt_mode"] = "cot_structured"
    elif "cot" in parts:
        cfg["prompt_mode"] = "cot"

    # Shuffle
    cfg["shuffle_choices"] = "noshuffle" not in parts

    # Paraphrases
    cfg["use_paraphrases"] = "nopara" not in parts

    # Scale (max_questions)
    if "100q" in parts:
        cfg["max_questions"] = 100

    return cfg if "prompt_mode" in cfg else None


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
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, valid["confidence"]))
    except Exception:
        return None


def compute_timing(run_data: dict, file_path: str | None = None) -> dict:
    """Compute elapsed time and progress."""
    cfg = run_data.get("config", {})
    # Use the fast byte-count if available (updated every refresh without
    # re-parsing the full file), fall back to actual result list length
    done_q = run_data.get("_fast_count") or len(run_data.get("question_results", []))

    # Determine total question count: max_questions if set, otherwise count
    # from the question database filtered by question_set
    total_q = cfg.get("max_questions")
    if total_q is None:
        qdb = load_question_db()
        question_set = cfg.get("question_set", "all")
        if question_set == "mmlu_standard":
            total_q = sum(1 for q in qdb.values() if q.get("variant") is None or q.get("variant") == "valid")
        elif question_set == "all":
            total_q = len(qdb)
        elif question_set == "broken_pairs":
            total_q = sum(1 for q in qdb.values() if q.get("pair_id") is not None)
        else:
            total_q = len(qdb) or done_q  # fallback
    pct = done_q / max(total_q, 1)

    # Compute elapsed time. Prefer the filename timestamp as the start time
    # because it's always set at file creation (even for resumed runs, where
    # the JSON "timestamp" field only covers the resume window).
    # Note: filename timestamps are local time (from datetime.now()), so we
    # use naive local time throughout to avoid timezone mismatches.
    elapsed_str = ""
    remaining_str = ""
    if file_path:
        try:
            fp = Path(file_path)
            mtime = fp.stat().st_mtime
            end = datetime.fromtimestamp(mtime)  # local time, no tz

            # Try to extract start time from filename: ..._YYYYMMDD_HHMMSS.json
            parts = fp.stem.split("_")
            start = None
            if len(parts) >= 2 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
                try:
                    start = datetime.strptime(
                        parts[-2] + "_" + parts[-1], "%Y%m%d_%H%M%S"
                    )  # local time, no tz — matches datetime.now() in pipeline
                except ValueError:
                    pass

            # Fall back to JSON timestamp if filename parsing fails
            if start is None:
                ts = run_data.get("timestamp", "")
                if ts:
                    start = datetime.fromisoformat(ts).replace(tzinfo=None)

            if start is not None:
                elapsed = (end - start).total_seconds()
                elapsed_str = _fmt_sec(elapsed)
                if pct >= 1:
                    remaining_str = "Done"
                elif elapsed > 0 and done_q > 0:
                    # To estimate remaining time, we need the rate of questions
                    # produced during THIS run, not the total in the file
                    # (which includes carried-over results from --resume).
                    # Track the first-observed count in session state so we
                    # can compute how many were added since the run started.
                    first_key = f"_first_count_{file_path}"
                    if first_key not in st.session_state:
                        st.session_state[first_key] = done_q
                    first_count = st.session_state[first_key]
                    produced = done_q - first_count

                    if produced > 0:
                        q_per_sec = produced / elapsed
                        remaining_q = max(total_q - done_q, 0)
                        remaining = remaining_q / q_per_sec
                        remaining_str = _fmt_sec(remaining)
                    else:
                        # First observation — can't estimate yet
                        remaining_str = "estimating…"
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

# Auto-refresh: uses streamlit-autorefresh component which triggers a clean
# browser-side reload, preserving widget state (unlike st.rerun which doesn't)
auto_refresh = st.sidebar.toggle("Auto-refresh (60s)", value=False)
if auto_refresh:
    st_autorefresh(interval=60_000, key="auto_refresh_counter")

# Build label → path mapping using lightweight config extraction (not full file load).
# Falls back to parsing the filename if extraction fails (e.g. file mid-write).
# NEVER loads the full file here — that's deferred to when the user selects a run.
label_to_path: dict[str, Path] = {}
for fp in result_files:
    cfg = _extract_config_from_file(str(fp))
    if cfg is None:
        # Fallback: infer config from filename pattern
        # e.g. "exp1_full_direct_shuffle_nopara_20260316_134902.json"
        stem = fp.stem
        cfg = _config_from_filename(stem)
    if cfg is None or not cfg:
        continue
    label = format_run_name(cfg)
    # Disambiguate if needed
    if label in label_to_path:
        label = f"{label} ({fp.stem})"
    label_to_path[label] = fp

available_labels = list(label_to_path.keys())

# Persist multiselect across auto-refreshes via session_state.
# Validate that previously selected runs are still in the available list
# (a run might temporarily vanish if its label changed, then come back).
if "selected_runs" not in st.session_state:
    st.session_state["selected_runs"] = available_labels[:1] if available_labels else []
else:
    valid = set(available_labels)
    st.session_state["selected_runs"] = [
        s for s in st.session_state["selected_runs"] if s in valid
    ]

selected_labels = st.sidebar.multiselect(
    "Select runs",
    options=available_labels,
    key="selected_runs",
)

# Build lightweight run metadata from config + byte-count (< 1s total).
# Full file parsing is deferred until a tab actually needs the data.
runs: list[dict] = []
for label in selected_labels:
    fp = label_to_path[label]

    # Config is already extracted for the sidebar label — reuse it
    cfg = _extract_config_from_file(str(fp)) or _config_from_filename(fp.stem) or {}
    stats = _fast_file_stats(str(fp))
    fast_count = stats["count"] if stats else 0

    # Build a minimal data dict for compute_timing (no full parse needed)
    light_data = {"config": cfg, "question_results": [], "_fast_count": fast_count}
    timing = compute_timing(light_data, str(fp))

    # Show metadata in sidebar
    st.sidebar.caption(
        f"**{label}**  \n"
        f"Model: {cfg.get('model', '?')} · Prompt: {cfg.get('prompt_mode', '?')} · "
        f"Shuffle: {'on' if cfg.get('shuffle_choices') else 'off'} · "
        f"Para: {'on' if cfg.get('use_paraphrases') else 'off'}  \n"
        f"Questions: {timing['done_q']}/{timing['total_q']}"
    )

    runs.append({
        "label": label,
        "path": fp,
        "config": cfg,
        "timing": timing,
        "stats": stats,
        # Full data loaded lazily by _get_run_data() below
    })

if not runs:
    if selected_labels:
        st.warning("Selected run(s) could not be loaded — the result file may be "
                    "mid-write. This should resolve on the next refresh.")
    else:
        st.info("Select at least one run to view results.")
    st.stop()


def _get_run_data(run: dict) -> dict | None:
    """Lazy-load the full result data for a run (cached after first call).

    Tabs that need the full question_results (Distributions, Comparison,
    Effects, Explorer) call this. The Progress tab doesn't need it at all.
    """
    if "raw" not in run:
        data = load_result_file(str(run["path"]))
        if data is None:
            return None
        run["raw"] = data
        run["df"] = build_dataframe(data)
        run["question_results"] = data.get("question_results", [])
    return run.get("raw")


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
        timing = run["timing"]
        stats = run.get("stats")

        # Accuracy and avg queries from fast byte-count stats
        if stats and (stats["correct"] + stats["incorrect"]) > 0:
            total_scored = stats["correct"] + stats["incorrect"]
            acc_str = f"{stats['correct'] / total_scored:.1%}"
        else:
            acc_str = "—"
        avg_q_str = f"{stats['avg_queries']:.1f}" if stats and stats["avg_queries"] else "—"

        st.markdown(f"### {run['label']}")
        cols = st.columns(5)
        cols[0].metric("Questions", f"{timing['done_q']}/{timing['total_q']}")
        cols[1].metric("Accuracy", acc_str)
        cols[2].metric("Avg Queries", avg_q_str)
        cols[3].metric("Elapsed", timing["elapsed_str"] or "—")
        cols[4].metric("Remaining", timing["remaining_str"] or "—")

        st.progress(timing["pct"])
        st.divider()


# ---------------------------------------------------------------------------
# Helper: ensure full data is loaded for tabs that need it
# ---------------------------------------------------------------------------

def _ensure_loaded(runs: list[dict]) -> list[dict]:
    """Trigger lazy-load for all runs and return only those that loaded OK."""
    loaded = []
    for run in runs:
        if _get_run_data(run) is not None:
            loaded.append(run)
    return loaded


# ---------------------------------------------------------------------------
# Tab 2: Probability Distributions
# ---------------------------------------------------------------------------

with tab2:
    st.markdown("# What does the model actually believe?")
    _ensure_loaded(runs)

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
                opacity=0.7, nbinsx=30, histnorm="percent",
            ))
        if conf_incorrect:
            fig.add_trace(go.Histogram(
                x=conf_incorrect, name="Incorrect", marker_color=GOLD,
                opacity=0.7, nbinsx=30, histnorm="percent",
            ))
        fig.update_layout(barmode="overlay")
        _fig_layout(fig, title=f"{run['label']} — Per-query max(prob)", height=300)
        fig.update_yaxes(title="% of group")
        st.plotly_chart(fig, width="stretch")

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
                opacity=0.7, nbinsx=20, histnorm="percent",
            ))
        if len(incorrect_ag):
            fig.add_trace(go.Histogram(
                x=incorrect_ag, name="Incorrect", marker_color=GOLD,
                opacity=0.7, nbinsx=20, histnorm="percent",
            ))
        fig.update_layout(barmode="overlay")
        _fig_layout(fig, title=f"{run['label']} — Agreement distribution", height=300)
        fig.update_xaxes(range=[0, 1.05])
        fig.update_yaxes(title="% of group")
        st.plotly_chart(fig, width="stretch")

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
            st.dataframe(display_df, width="stretch", hide_index=True)

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
                opacity=0.7, nbinsx=25, histnorm="percent",
            ))
        if len(incorrect_conf):
            fig.add_trace(go.Histogram(
                x=incorrect_conf, name="Incorrect", marker_color=GOLD,
                opacity=0.7, nbinsx=25, histnorm="percent",
            ))
        fig.update_layout(barmode="overlay")
        _fig_layout(fig, title=f"{run['label']} — max(mean_probs)", height=300)
        fig.update_yaxes(title="% of group")
        st.plotly_chart(fig, width="stretch")

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
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Condition Comparison
# ---------------------------------------------------------------------------

with tab3:
    st.markdown("# Condition Comparison")
    _ensure_loaded(runs)

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
            colorscale=[[0, GOLD], [0.5, SLATE], [1, TEAL]],
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(title="Accuracy"),
        ))
        _fig_layout(fig, height=max(200, 60 * len(runs) + 80))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

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
            st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4: Effect Analysis
# ---------------------------------------------------------------------------

def _extract_condition(cfg: dict) -> dict:
    """Extract condition flags from a run config."""
    return {
        "think": bool(cfg.get("think")),
        "prompt_mode": cfg.get("prompt_mode", "direct"),
        "shuffle": bool(cfg.get("shuffle_choices", False)),
        "para": bool(cfg.get("use_paraphrases", False)),
    }


def _condition_label(cond: dict, exclude_var: str | None = None) -> str:
    """Human-readable label from condition flags, excluding one variable."""
    parts = []
    if exclude_var != "think":
        parts.append("think on" if cond["think"] else "think off")
    if exclude_var != "prompt_mode":
        pm = cond.get("prompt_mode", "direct")
        pm_label = {"direct": "direct", "cot": "CoT", "cot_structured": "CoT-struct"}.get(pm, pm)
        parts.append(pm_label)
    if exclude_var != "shuffle":
        parts.append("shuffle on" if cond["shuffle"] else "shuffle off")
    if exclude_var != "para":
        parts.append("para on" if cond["para"] else "para off")
    return " · ".join(parts)


def _compute_run_metrics(run: dict) -> dict:
    """Compute key metrics for a single run."""
    df = run["df"]
    valid = df[df["correct"].notna()]
    acc = valid["correct"].mean() if len(valid) > 0 else None
    multi = df[df["agreement"].notna()]
    avg_agreement = float(multi["agreement"].mean()) if len(multi) > 0 else None
    avg_confidence = float(df["confidence"].mean()) if len(df) > 0 else None
    return {
        "accuracy": acc,
        "auroc": compute_auroc(df),
        "confidence": avg_confidence,
        "agreement": avg_agreement,
    }


with tab4:
    st.markdown("# Effect Analysis")
    _ensure_loaded(runs)
    st.caption("What does each experimental variable actually do?")

    if len(runs) < 2:
        st.info("Select two or more runs to analyse effects.")
    else:
        # --- Index runs by their condition tuple ---
        run_by_condition: dict[tuple, dict] = {}
        for run in runs:
            cond = _extract_condition(run["config"])
            key = (cond["think"], cond["prompt_mode"], cond["shuffle"], cond["para"])
            run_by_condition[key] = run

        # --- Matched pair finders ---
        BINARY_VARS = [
            ("think", "Think on vs off", 0),
            ("shuffle", "Shuffle on vs off", 2),
            ("para", "Para on vs off", 3),
        ]

        def _find_binary_matched_pairs(var_name: str, key_idx: int) -> list[tuple[dict, dict, dict]]:
            """Find matched pairs differing only on a binary variable.

            Returns (cond_off, run_off, run_on) tuples.
            """
            pairs = []
            seen = set()
            for key, run in run_by_condition.items():
                flipped = list(key)
                flipped[key_idx] = not flipped[key_idx]
                flipped_key = tuple(flipped)
                pair_id = tuple(sorted([key, flipped_key]))
                if pair_id in seen:
                    continue
                seen.add(pair_id)
                if flipped_key in run_by_condition:
                    if key[key_idx]:
                        run_on, run_off = run, run_by_condition[flipped_key]
                        cond_off = _extract_condition(run_by_condition[flipped_key]["config"])
                    else:
                        run_off, run_on = run, run_by_condition[flipped_key]
                        cond_off = _extract_condition(run["config"])
                    pairs.append((cond_off, run_off, run_on))
            return pairs

        def _find_prompt_matched_pairs(target_mode: str) -> list[tuple[dict, dict, dict]]:
            """Find matched pairs: direct vs target_mode."""
            pairs = []
            seen = set()
            for key, run in run_by_condition.items():
                if key[1] != "direct":
                    continue
                target_key = (key[0], target_mode, key[2], key[3])
                pair_id = tuple(sorted([key, target_key]))
                if pair_id in seen:
                    continue
                seen.add(pair_id)
                if target_key in run_by_condition:
                    cond_direct = _extract_condition(run["config"])
                    pairs.append((cond_direct, run, run_by_condition[target_key]))
            return pairs

        # Build variable list dynamically based on what's loaded
        VARIABLES: list[tuple[str, str]] = [
            ("think", "Think on vs off"),
            ("shuffle", "Shuffle on vs off"),
            ("para", "Para on vs off"),
        ]

        _prompt_modes_present = {key[1] for key in run_by_condition}
        if "cot" in _prompt_modes_present:
            VARIABLES.append(("prompt_cot", "Direct vs CoT"))
        if "cot_structured" in _prompt_modes_present:
            VARIABLES.append(("prompt_cotstruct", "Direct vs CoT-struct"))

        def _find_matched_pairs(var_name: str) -> list[tuple[dict, dict, dict]]:
            """Dispatch to the right pair-finder for a given variable."""
            for bvar, _, kidx in BINARY_VARS:
                if bvar == var_name:
                    return _find_binary_matched_pairs(var_name, kidx)
            if var_name == "prompt_cot":
                return _find_prompt_matched_pairs("cot")
            if var_name == "prompt_cotstruct":
                return _find_prompt_matched_pairs("cot_structured")
            return []

        # --- Section 1: Main effects table ---
        st.markdown("### Main effects")
        st.caption(
            "Each delta is the average across all matched pairs that differ "
            "only on that variable. Positive accuracy/AUROC = better. "
            "Confidence and agreement show how the variable shifts model behaviour."
        )

        # Metrics config: (key, display_name, format, higher_is_better)
        METRICS = [
            ("accuracy", "Accuracy Δ", lambda x: f"{x:+.1%}", True),
            ("auroc", "AUROC Δ", lambda x: f"{x:+.3f}", True),
            ("confidence", "Confidence Δ", lambda x: f"{x:+.3f}", None),
            ("agreement", "Agreement Δ", lambda x: f"{x:+.3f}", None),
        ]

        effect_rows = []
        all_pair_data: dict[str, list] = {}

        for var_name, var_label in VARIABLES:
            pairs = _find_matched_pairs(var_name)
            if not pairs:
                effect_rows.append({
                    "Effect": var_label, "Pairs": 0,
                    **{m[1]: None for m in METRICS},
                })
                all_pair_data[var_name] = []
                continue

            deltas = {m[0]: [] for m in METRICS}
            pair_details = []

            for cond_off, run_off, run_on in pairs:
                m_off = _compute_run_metrics(run_off)
                m_on = _compute_run_metrics(run_on)
                pair_info = {
                    "label": _condition_label(cond_off, exclude_var=var_name),
                    "deltas": {},
                }
                for metric_key, _, _, _ in METRICS:
                    v_off = m_off[metric_key]
                    v_on = m_on[metric_key]
                    if v_off is not None and v_on is not None:
                        d = v_on - v_off
                        deltas[metric_key].append(d)
                        pair_info["deltas"][metric_key] = d
                pair_details.append(pair_info)

            all_pair_data[var_name] = pair_details

            def _mean_or_none(lst):
                return sum(lst) / len(lst) if lst else None

            row = {"Effect": var_label, "Pairs": len(pairs)}
            for metric_key, col_name, _, _ in METRICS:
                row[col_name] = _mean_or_none(deltas[metric_key])
            effect_rows.append(row)

        if effect_rows:
            eff_df = pd.DataFrame(effect_rows)

            def _colour_delta(val, higher_is_better: bool | None) -> str:
                """Teal for favourable, gold for unfavourable, neutral for informational."""
                if pd.isna(val) or val is None or higher_is_better is None:
                    return ""
                if higher_is_better:
                    return f"color: {TEAL}" if val > 0 else (f"color: {GOLD}" if val < 0 else "")
                else:
                    return f"color: {TEAL}" if val < 0 else (f"color: {GOLD}" if val > 0 else "")

            def _style_effects(sdf: pd.DataFrame) -> pd.io.formats.style.Styler:
                fmt = {m[1]: (lambda x, f=m[2]: f(x) if pd.notna(x) else "—") for m in METRICS}
                fmt["Pairs"] = lambda x: str(x)
                styled = sdf.style.format(fmt)
                for _, col_name, _, hib in METRICS:
                    styled = styled.map(
                        lambda v, h=hib: _colour_delta(v, h), subset=[col_name],
                    )
                return styled

            st.dataframe(_style_effects(eff_df), width="stretch", hide_index=True)

            missing_vars = [row["Effect"] for row in effect_rows if row["Pairs"] == 0]
            if missing_vars:
                st.caption(
                    f"No matched pairs found for: {', '.join(missing_vars)}. "
                    "Load more runs with contrasting conditions."
                )

        st.markdown(
            "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
            unsafe_allow_html=True,
        )

        # --- Section 2: Effect consistency bar charts ---
        st.markdown("### Effect consistency")
        st.caption(
            "Each bar is one matched pair. If all bars point the same direction, "
            "the effect is robust across conditions."
        )

        for metric_key, metric_label, metric_fmt, _ in METRICS[:2]:
            # Show consistency charts for accuracy and AUROC
            st.markdown(f"**{metric_label.replace(' Δ', '')} deltas by variable**")
            cols = st.columns(max(len(VARIABLES), 1))

            for col_idx, (var_name, var_label) in enumerate(VARIABLES):
                with cols[col_idx]:
                    pair_details = all_pair_data.get(var_name, [])
                    if not pair_details:
                        st.caption(f"{var_label}: no pairs")
                        continue

                    labels = []
                    values = []
                    colors = []
                    for pd_item in pair_details:
                        d = pd_item["deltas"].get(metric_key)
                        if d is not None:
                            labels.append(pd_item["label"])
                            values.append(d)
                            colors.append(TEAL if d >= 0 else GOLD)

                    if not values:
                        st.caption(f"{var_label}: no data")
                        continue

                    text_vals = [metric_fmt(v) for v in values]
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        y=labels, x=values, orientation="h",
                        marker_color=colors, text=text_vals,
                        textposition="outside", textfont=dict(size=11),
                    ))
                    fig_bar.add_vline(x=0, line_color=GRAY_LIGHT, line_width=1)
                    _fig_layout(fig_bar, title=var_label,
                                height=max(180, 60 * len(values)))
                    fig_bar.update_layout(showlegend=False,
                                          yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_bar, width="stretch")

        st.markdown(
            "<hr style='border:none; border-top:1px solid #E5E0DB; margin:16px 0;'>",
            unsafe_allow_html=True,
        )

        # --- Section 3: Interaction spotlight ---
        st.markdown("### Interaction spotlight")
        st.caption(
            "Flags variables whose effect on accuracy flips sign depending on "
            "other conditions — a sign of interaction between variables."
        )

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

            signs = [d >= 0 for _, d in acc_deltas]
            if len(set(signs)) > 1:
                pos_pairs = [(lbl, d) for lbl, d in acc_deltas if d >= 0]
                neg_pairs = [(lbl, d) for lbl, d in acc_deltas if d < 0]
                best_pos = max(pos_pairs, key=lambda x: x[1]) if pos_pairs else None
                worst_neg = min(neg_pairs, key=lambda x: x[1]) if neg_pairs else None

                short_name = var_label.split(" on")[0].split(" vs")[0]
                msg = f"**{short_name}** effect on accuracy flips sign: "
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
                "All effects are consistent across conditions — no sign-flip "
                "interactions detected."
            )


# ---------------------------------------------------------------------------
# Tab 5: Question Explorer
# ---------------------------------------------------------------------------

with tab5:
    st.markdown("# Question Explorer")
    _ensure_loaded(runs)

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
        correct_color = TEAL if correct else (GOLD if correct is False else GRAY_MID)

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
                table_rows.append({
                    "Query text": ql.get("query_text", ""),
                    "Answer": ANSWER_LETTERS[ql.get("canonical_answer", 0)],
                    "Probs": prob_display,
                })
            st.dataframe(
                pd.DataFrame(table_rows),
                width="stretch", hide_index=True,
                column_config={"Query text": st.column_config.TextColumn(width="large")},
            )

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
            st.plotly_chart(fig, width="stretch")

        st.divider()
