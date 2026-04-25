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
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import betainc as _betainc, digamma as _digamma
from scipy.stats import norm as _norm
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SIGNALS_CSV = PROJECT_ROOT / "analysis" / "signals.csv"
ANALYSIS_CACHE = DASHBOARD_DIR / "analysis_cache.pkl.gz"

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
MODEL_COLORS = {"Qwen 3": TEAL, "Gemma 4": ROSE, "Qwen 3.5": GOLD}
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
    border-radius: 6px; padding: 5px 8px; }
[data-testid="stMetricLabel"] { font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.04em; color: #8B95A1; }
[data-testid="stMetricValue"] { font-size: 16px; font-weight: 400; }
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


def _effective_mode(cfg: dict) -> str:
    """Map prompt_mode + think to a single mode label."""
    if cfg.get("think", False):
        return "think"
    return "CoT" if cfg.get("prompt_mode", "direct") == "cot" else "direct"


def _short_model_name(cfg: dict) -> str:
    """Extract a short model label from config."""
    raw = cfg.get("model_name", "") or cfg.get("run_name", "")
    low = raw.lower()
    if "gemma4" in low or "gemma-4" in low:
        return "Gemma 4"
    if "qwen3.5" in low or "qwen3-5" in low or "qwen35" in low:
        return "Qwen 3.5"
    if "qwen3" in low or "qwen-3" in low:
        return "Qwen 3"
    return raw.split(":")[0] if ":" in raw else raw.split("_")[0]


def format_run_label(cfg: dict) -> str:
    """Human-readable run label.  Only call out positives (shuffle, paraphrase)."""
    parts = [_short_model_name(cfg)]
    parts.append(_effective_mode(cfg))
    if cfg.get("shuffle_options", False):
        parts.append("shuffle")
    if cfg.get("use_paraphrases", False):
        parts.append("paraphrase")
    return " · ".join(parts)


def _sample_inference_times(file_path: Path, tail_bytes: int = 65536) -> list[float]:
    """Extract recent inference_time_s values from the tail of a result JSON."""
    import re as _re
    try:
        size = file_path.stat().st_size
        with open(file_path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            chunk = f.read().decode("utf-8", errors="ignore")
        return [float(m) for m in _re.findall(r'"inference_time_s":\s*([\d.]+)', chunk)]
    except Exception:
        return []


def compute_timing(file_path: Path, done_q: int, total_q: int,
                    completed_at: str | None = None,
                    num_permutations: int = 1) -> dict:
    """Compute ETA from per-query inference_time_s recorded in result files.

    Primary: reads recent inference times from the file tail, computes mean
    time per question (mean_query_time × num_permutations), extrapolates.
    Fallback: filename timestamp + question count for runs without timing data.
    """
    elapsed_str = ""
    remaining_str = ""
    rate_str = ""
    pct = min(done_q / max(total_q, 1), 1.0)
    finished = done_q >= total_q

    try:
        if finished and completed_at:
            parts = file_path.stem.split("_")
            if len(parts) >= 2 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
                start = datetime.strptime(parts[-2] + "_" + parts[-1], "%Y%m%d_%H%M%S")
                end = datetime.fromisoformat(completed_at).replace(tzinfo=None)
                elapsed_str = _fmt_sec((end - start).total_seconds())
            remaining_str = "Done"
        elif finished:
            remaining_str = "Done"
        else:
            times = _sample_inference_times(file_path)
            if times:
                mean_query_time = sum(times) / len(times)
                time_per_question = mean_query_time * num_permutations
                remaining_q = total_q - done_q
                remaining_sec = remaining_q * time_per_question
                rate_q_per_min = 60.0 / time_per_question if time_per_question > 0 else 0
                remaining_str = f"~{_fmt_sec(remaining_sec)}"
                rate_str = f"{rate_q_per_min:.1f} q/min"
            else:
                parts = file_path.stem.split("_")
                if len(parts) >= 2 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
                    start = datetime.strptime(parts[-2] + "_" + parts[-1], "%Y%m%d_%H%M%S")
                    elapsed_sec = (datetime.now() - start).total_seconds()
                    if done_q > 0 and elapsed_sec > 0:
                        fallback_rate = done_q / elapsed_sec
                        remaining_sec = (total_q - done_q) / fallback_rate
                        remaining_str = f"~{_fmt_sec(remaining_sec)}*"
                        rate_str = f"{fallback_rate * 60:.1f} q/min*"
                if not remaining_str:
                    remaining_str = "measuring..."
    except Exception:
        pass

    return {"pct": pct, "elapsed": elapsed_str, "remaining": remaining_str, "rate": rate_str}


def results_to_df(all_data: dict[str, dict]) -> pd.DataFrame:
    """Convert loaded result dicts to a flat DataFrame.

    Deduplicates by (run_name, question_id) — keeps first occurrence.
    This handles the case where a resumed run appended duplicate results.
    """
    rows = []
    seen: set[tuple[str, str]] = set()
    for run_name, data in all_data.items():
        cfg = data.get("config", {})
        model = _short_model_name(cfg)
        context = cfg.get("context_condition", "unknown")
        think = cfg.get("think", False)
        prompt_mode = cfg.get("prompt_mode", "direct")
        shuffle = cfg.get("shuffle_options", True)
        paraphrase = cfg.get("use_paraphrases", False)

        for qr in data.get("question_results", []):
            if qr.get("skipped", False) or not qr.get("mean_probs"):
                continue

            # Deduplicate: skip if we've already seen this question in this run
            qid = qr.get("question_id", "")
            dedup_key = (run_name, qid)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            query_log = qr.get("query_log", [])
            n_queries = len(query_log)
            mean_probs = qr.get("mean_probs", [0.25] * 4)
            final_answer = qr.get("final_answer", 0)

            query_probs = [ql.get("canonical_probs", [0.25]*4) for ql in query_log
                           if len(ql.get("canonical_probs", [])) == 4]
            if query_probs:
                per_query_argmax = [int(np.argmax(p)) for p in query_probs]
                agreement = sum(1 for a in per_query_argmax if a == final_answer) / len(per_query_argmax)

                # Epistemic/aleatoric decomposition
                qp = np.array(query_probs)
                qp = np.clip(qp, 1e-12, None)  # avoid log(0)
                mean_dist = qp.mean(axis=0)
                total_entropy = -float(np.sum(mean_dist * np.log(mean_dist)))
                per_query_entropy = -np.sum(qp * np.log(qp), axis=1)
                aleatoric = float(per_query_entropy.mean())
                epistemic = total_entropy - aleatoric
            else:
                agreement = float("nan")
                total_entropy = float("nan")
                aleatoric = float("nan")
                epistemic = float("nan")

            mode = _effective_mode(cfg)

            rows.append({
                "run_name": run_name,
                "model": model,
                "context_condition": context,
                "think": think,
                "prompt_mode": prompt_mode,
                "mode": mode,
                "shuffle": shuffle,
                "paraphrase": paraphrase,
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
                "single_entropy": -float(np.sum(np.array(mean_probs).clip(1e-12) * np.log(np.array(mean_probs).clip(1e-12)))) if mean_probs else float("nan"),
                "agreement": agreement,
                "total_entropy": total_entropy,
                "aleatoric": aleatoric,
                "epistemic": epistemic,
                "query_log": query_log,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _slim_query_log(query_log: list[dict]) -> list[dict]:
    """Strip heavy text fields from query_log entries for caching."""
    keep = ("canonical_probs", "canonical_answer", "pass1_canonical_answer",
            "answer_permutation", "inference_time_s", "query_number")
    return [{k: ql.get(k) for k in keep if k in ql} for ql in query_log]


def build_analysis_cache() -> dict:
    """Build a compact cache of all analysis data from result files."""
    import pickle as _pickle
    import gzip as _gzip

    files = get_result_files()
    all_data: dict[str, dict] = {}
    run_meta: dict[str, dict] = {}
    for fp in files:
        data = _load_json(fp)
        if data:
            prefix = _extract_run_prefix(fp.stem)
            all_data[prefix] = data
            cfg = data.get("config", {})
            run_meta[prefix] = {
                "config": cfg,
                "label": format_run_label(cfg) if cfg else prefix,
            }

    full_df = results_to_df(all_data)

    if not full_df.empty:
        if "query_log" in full_df.columns:
            full_df["query_log"] = full_df["query_log"].apply(_slim_query_log)
        full_df = full_df.drop(columns=["question_text", "options"], errors="ignore")

    signals = load_signals()

    adaptive_data = None
    _adaptive_path = DASHBOARD_DIR / "adaptive_cache.pkl"
    if _adaptive_path.exists():
        try:
            with open(_adaptive_path, "rb") as _af:
                adaptive_data = _pickle.load(_af)
        except Exception:
            pass

    cache = {
        "df": full_df,
        "run_meta": run_meta,
        "signals_df": signals,
        "adaptive": adaptive_data,
        "built_at": datetime.now().isoformat(),
    }

    raw = _pickle.dumps(cache, protocol=4)
    with open(ANALYSIS_CACHE, "wb") as f:
        f.write(_gzip.compress(raw, compresslevel=6))

    return cache


@st.cache_data(ttl=None)
def _load_analysis_cache(mtime: float) -> dict | None:
    import pickle as _pickle
    import gzip as _gzip
    try:
        with open(ANALYSIS_CACHE, "rb") as f:
            return _pickle.load(_gzip.open(f, "rb"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("# Pre-Action UQ")

result_files = get_result_files()
HAS_LOCAL_FILES = len(result_files) > 0
HAS_ANALYSIS_CACHE = ANALYSIS_CACHE.exists()

if not HAS_LOCAL_FILES and not HAS_ANALYSIS_CACHE:
    st.sidebar.warning("No result files or analysis cache found")
    st.title("Pre-Action Uncertainty Quantification")
    st.info("No results found. Run experiments first, or deploy with an analysis cache.")
    st.stop()

# Auto-refresh — only touches Progress tab (live file reading)
auto_refresh = False
if HAS_LOCAL_FILES:
    auto_refresh = st.sidebar.toggle("Auto-refresh progress (2 min)", value=False)
    if auto_refresh:
        st_autorefresh(interval=120_000, key="auto_refresh_counter")

# --- Cache management buttons ---
st.sidebar.divider()

if HAS_LOCAL_FILES:
    if st.sidebar.button("Update Analysis Cache", key="update_analysis"):
        with st.sidebar.status("Building analysis cache...", expanded=True):
            t0 = time.time()
            cache = build_analysis_cache()
            elapsed = time.time() - t0
            n_runs = len(cache.get("run_meta", {}))
            n_rows = len(cache["df"]) if cache["df"] is not None and not cache["df"].empty else 0
            sz = ANALYSIS_CACHE.stat().st_size / 1e6
            st.sidebar.success(
                f"Cache built in {elapsed:.0f}s — {n_runs} runs, "
                f"{n_rows:,} questions, {sz:.1f} MB"
            )
        _load_analysis_cache.clear()
        st.rerun()

    st.sidebar.caption("Adaptive cache: ~36 min to recompute")
    if st.sidebar.button("Recalculate Adaptive Cache", key="recalc_adaptive"):
        import subprocess, sys
        _script = Path(__file__).resolve().parent / "precompute_adaptive.py"
        _python = Path(sys.executable)
        with st.sidebar.status("Recomputing adaptive sampling...", expanded=True):
            _proc = subprocess.run(
                [str(_python), str(_script)],
                capture_output=True, text=True, timeout=3600,
            )
            if _proc.returncode == 0:
                st.sidebar.success("Cache updated! Refresh the page to use new results.")
            else:
                st.sidebar.error(f"Precompute failed:\n{_proc.stderr[-500:]}")
            if _proc.stdout:
                st.sidebar.text(_proc.stdout[-1000:])

# --- Load analysis data from cache ---
analysis_cache = None
if HAS_ANALYSIS_CACHE:
    _cache_mtime = ANALYSIS_CACHE.stat().st_mtime
    analysis_cache = _load_analysis_cache(_cache_mtime)

_run_name_cfg: dict[str, dict] = {}

if analysis_cache is not None:
    df = analysis_cache["df"]
    signals_df = analysis_cache.get("signals_df")
    _cached_meta = analysis_cache.get("run_meta", {})
    _cache_time = analysis_cache.get("built_at", "unknown")
    st.sidebar.caption(f"Analysis cache: {_cache_time}")

    for prefix, meta in _cached_meta.items():
        _run_name_cfg[prefix] = meta.get("config", {})

    if HAS_LOCAL_FILES:
        # Build real file paths for progress tab, cache for everything else
        run_labels: dict[str, Path] = {}
        for fp in result_files:
            cfg = _extract_config_head(fp)
            if cfg:
                label = format_run_label(cfg)
            else:
                label = _extract_run_prefix(fp.stem).replace("quality_pilot_", "")
            run_labels[label] = fp
        selected_paths: dict[str, Path | None] = run_labels.copy()
    else:
        run_labels = {v["label"]: None for v in _cached_meta.values()}
        selected_paths: dict[str, Path | None] = run_labels.copy()
    selected_labels = list(run_labels.keys())
else:
    # No cache — build df live from files (first run / cache deleted)
    run_labels: dict[str, Path] = {}
    for fp in result_files:
        cfg = _extract_config_head(fp)
        if cfg:
            label = format_run_label(cfg)
        else:
            label = _extract_run_prefix(fp.stem).replace("quality_pilot_", "")
        run_labels[label] = fp
        if cfg:
            _run_name_cfg[_extract_run_prefix(fp.stem)] = cfg

    selected_labels = list(run_labels.keys())
    selected_paths = run_labels.copy()

    all_data: dict[str, dict] = {}
    for label, fp in selected_paths.items():
        data = _load_json(fp)
        if data:
            prefix = _extract_run_prefix(fp.stem)
            all_data[prefix] = data

    df = results_to_df(all_data)
    signals_df = load_signals()
    st.sidebar.warning("No analysis cache — click 'Update Analysis Cache' to build one.")


def _cfg_for_run_name(run_name: str) -> dict | None:
    """Look up config for a run_name, using cache metadata or file-based fallback."""
    if run_name in _run_name_cfg:
        return _run_name_cfg[run_name]
    for _l, _p in selected_paths.items():
        if _p is not None and _extract_run_prefix(_p.stem) == run_name:
            return _extract_config_head(_p)
    return None


# ---------------------------------------------------------------------------
# Tab 1: Progress
# ---------------------------------------------------------------------------

def _progress_row(label: str, fp: Path) -> dict:
    """Compute progress info for a single run. Returns dict with display data."""
    cfg = _extract_config_head(fp)
    stats = _fast_file_stats(fp)
    total_q = (cfg.get("max_questions") if cfg else None) or 4609
    done_q = stats["count"]
    correct = stats["correct"]
    incorrect = stats["incorrect"]
    acc = correct / max(correct + incorrect, 1)
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
    num_perm = cfg.get("num_permutations", 1) if cfg else 1
    timing = compute_timing(fp, done_q, total_q, completed_at=completed_at,
                            num_permutations=num_perm)
    skipped = done_q - correct - incorrect
    is_done = done_q >= total_q or completed_at is not None
    return {
        "label": label, "done_q": done_q, "total_q": total_q,
        "acc": acc, "skipped": skipped, "timing": timing, "is_done": is_done,
    }


def _render_progress_row(row: dict) -> None:
    """Render a single compact progress row."""
    col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1], gap="small")
    with col1:
        st.markdown(f"<p style='font-size:13px;font-weight:500;margin:0 0 4px 0'>{row['label']}</p>",
                    unsafe_allow_html=True)
        st.progress(row["timing"]["pct"])
    with col2:
        st.metric("Questions", f"{row['done_q']}/{row['total_q']}")
    with col3:
        st.metric("Accuracy", f"{row['acc']:.0%}")
    with col4:
        st.metric("Skipped", row["skipped"] if row["skipped"] > 0 else "-")
    with col5:
        st.metric("Rate", row["timing"].get("rate") or "-")
    with col6:
        st.metric("ETA", row["timing"]["remaining"] or "-")
    st.markdown("<hr style='margin:4px 0;border:none;border-top:1px solid #E5E0DB'>",
                unsafe_allow_html=True)


def tab_progress() -> None:
    st.header("Experiment Progress")

    if not selected_paths:
        st.info("Select at least one run.")
        return

    rows = [_progress_row(label, fp) for label, fp in selected_paths.items() if fp is not None]
    in_progress = [r for r in rows if not r["is_done"]]
    completed = [r for r in rows if r["is_done"]]

    if in_progress:
        st.subheader(f"In Progress ({len(in_progress)})")
        for row in in_progress:
            _render_progress_row(row)

    if completed:
        st.subheader(f"Completed ({len(completed)})")
        for row in completed:
            _render_progress_row(row)


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
        cfg = _cfg_for_run_name(run_name)
        return format_run_label(cfg) if cfg else run_name.replace("quality_", "")

    def _run_sort_key(run_name):
        """Sort runs into visual groups: mode, then shuffle/para variant, then model."""
        cfg = _cfg_for_run_name(run_name)
        if cfg:
            mode = _effective_mode(cfg)
            shuffle = cfg.get("shuffle_options", False)
            para = cfg.get("use_paraphrases", False)
            model = _short_model_name(cfg)
        else:
            mode, shuffle, para, model = "direct", False, False, ""
        mode_order = {"direct": 0, "CoT": 1, "think": 2}
        variant_order = (0 if not shuffle and not para else
                         1 if shuffle and not para else
                         2 if not shuffle and para else 3)
        return (mode_order.get(mode, 9), variant_order, model)

    def _run_group_label(run_name):
        """Group header like 'Direct' or 'Direct · shuffle'."""
        cfg = _cfg_for_run_name(run_name)
        if cfg:
            mode = _effective_mode(cfg)
            shuffle = cfg.get("shuffle_options", False)
            para = cfg.get("use_paraphrases", False)
        else:
            mode, shuffle, para = "direct", False, False
        parts = [mode.capitalize() if mode != "CoT" else "CoT"]
        if shuffle:
            parts.append("shuffle")
        if para:
            parts.append("paraphrase")
        return " · ".join(parts)

    # --- Per-model MSP distributions (grouped by model, subplots per config) ---
    st.subheader("MSP by Condition (Correct vs Incorrect)")
    st.caption("Teal = correct, Gold = incorrect. Grouped by model to compare across conditions.")
    run_names = sorted(active["run_name"].unique(), key=_run_sort_key)

    models_in_data = []
    runs_by_model: dict[str, list[str]] = {}
    for rn in run_names:
        cfg = _cfg_for_run_name(rn)
        model = _short_model_name(cfg) if cfg else "Unknown"
        if model not in runs_by_model:
            runs_by_model[model] = []
            models_in_data.append(model)
        runs_by_model[model].append(rn)

    for model in models_in_data:
        st.markdown(f"---")
        st.markdown(f"**{model}**")
        mruns = runs_by_model[model]
        for i in range(0, len(mruns), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(mruns):
                    break
                sub = active[active["run_name"] == mruns[idx]]
                with col:
                    st.plotly_chart(_round_hover(_msp_hist(sub, _run_group_label(mruns[idx]))),
                                    width="stretch")

    # --- Agreement (only for multi-query runs, grouped by model) ---
    multi = active[(active["num_queries"] > 1) & active["agreement"].notna()]
    if not multi.empty:
        st.subheader("Agreement by Condition (Correct vs Incorrect)")
        st.caption("Grouped by model to compare how shuffle/paraphrase affects agreement.")
        multi_runs = sorted(multi["run_name"].unique(), key=_run_sort_key)

        mruns_by_model: dict[str, list[str]] = {}
        mmodels: list[str] = []
        for rn in multi_runs:
            cfg = _cfg_for_run_name(rn)
            model = _short_model_name(cfg) if cfg else "Unknown"
            if model not in mruns_by_model:
                mruns_by_model[model] = []
                mmodels.append(model)
            mruns_by_model[model].append(rn)

        for model in mmodels:
            st.markdown(f"---")
            st.markdown(f"**{model}**")
            mbuf = mruns_by_model[model]
            for i in range(0, len(mbuf), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(mbuf):
                        break
                    sub = multi[multi["run_name"] == mbuf[idx]]
                    with col:
                        st.plotly_chart(_round_hover(_agree_hist(sub, _run_group_label(mbuf[idx]))),
                                        width="stretch")

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

        fc_runs = sorted(
            [rn for rn in multi_fc["run_name"].unique()
             if len(multi_fc[multi_fc["run_name"] == rn]) >= 5],
            key=_run_sort_key,
        )
        fc_by_model: dict[str, list[str]] = {}
        fc_models: list[str] = []
        for rn in fc_runs:
            cfg = _cfg_for_run_name(rn)
            model = _short_model_name(cfg) if cfg else "Unknown"
            if model not in fc_by_model:
                fc_by_model[model] = []
                fc_models.append(model)
            fc_by_model[model].append(rn)

        for model in fc_models:
            st.markdown(f"**{model}**")
            mbuf = fc_by_model[model]
            for i in range(0, len(mbuf), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(mbuf):
                        break
                    sub = multi_fc[multi_fc["run_name"] == mbuf[idx]].dropna(subset=["agreement", "msp"])
                    with col:
                        st.plotly_chart(_round_hover(_fragile_box(sub, _run_group_label(mbuf[idx]))),
                                        width="stretch")
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
    # Model is always shown first as the primary grouping dimension
    factor_cols = ["model"] if "model" in active.columns else []
    factor_labels = {
        "model": "Model",
        "mode": "Mode",
        "shuffle": "Shuffle",
        "paraphrase": "Paraphrase",
    }
    for col, label in factor_labels.items():
        if col == "model":
            continue  # already added above
        if col in active.columns and active[col].nunique() > 1:
            factor_cols.append(col)

    # If no factors vary, just show model
    if not factor_cols:
        factor_cols = ["model"] if "model" in active.columns else ["mode"]

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
        row["_acc_sort"] = acc
        pivot_data.append(row)

    # Sort by overall accuracy descending
    pivot_data.sort(key=lambda r: r.pop("_acc_sort", 0), reverse=True)

    if pivot_data:
        # Build HTML table with grouped column headers
        factor_hdrs = [factor_labels.get(c, c) for c in factor_cols]
        n_factors = len(factor_hdrs)

        # Per-column best/worst accuracy highlighting
        acc_extremes: dict[str, tuple[float, float, float]] = {}
        for key in ("Accuracy", "Acc Easy", "Acc Hard"):
            vals = []
            for row in pivot_data:
                v = row.get(key, "-")
                if v != "-":
                    try:
                        vals.append(float(v.replace("%", "")))
                    except ValueError:
                        pass
            if len(vals) < 3:
                continue
            lo, hi = min(vals), max(vals)
            acc_extremes[key] = (lo, hi, hi - lo)

        def _acc_highlight(val_str: str, col: str) -> str:
            ext = acc_extremes.get(col)
            if not ext or ext[2] < 1.0:
                return ""
            try:
                v = float(val_str.replace("%", ""))
            except (ValueError, AttributeError):
                return ""
            if abs(v - ext[1]) < 0.05:
                return ' style="background:rgba(42,140,143,0.20);font-weight:700"'
            if abs(v - ext[0]) < 0.05 and ext[2] >= 3.0:
                return ' style="background:rgba(202,74,122,0.12)"'
            return ""

        # Confidence calibration: top 3 best calibrated (teal) + top 3 worst overconfident (rose)
        conf_styles: dict[tuple[str, int], str] = {}
        for conf_key, acc_key in [("Confidence", "Accuracy"), ("Conf Easy", "Acc Easy"), ("Conf Hard", "Acc Hard")]:
            gaps = []
            for ri, row in enumerate(pivot_data):
                a, c = row.get(acc_key, "-"), row.get(conf_key, "-")
                if a == "-" or c == "-":
                    continue
                try:
                    gaps.append((ri, float(c) - float(a.replace("%", "")) / 100))
                except ValueError:
                    pass
            if len(gaps) < 5:
                continue
            by_abs = sorted(gaps, key=lambda x: abs(x[1]))
            for rank, (ri, _) in enumerate(by_abs[:3]):
                style = 'background:rgba(42,140,143,0.20);font-weight:700' if rank == 0 else 'background:rgba(42,140,143,0.12)'
                conf_styles[(conf_key, ri)] = style
            by_overconf = sorted(gaps, key=lambda x: x[1], reverse=True)
            for rank, (ri, _) in enumerate(by_overconf[:3]):
                if (conf_key, ri) in conf_styles:
                    continue
                style = 'background:rgba(202,74,122,0.18);font-weight:700' if rank == 0 else 'background:rgba(202,74,122,0.10)'
                conf_styles[(conf_key, ri)] = style

        def _conf_highlight(conf_key: str, row_idx: int) -> str:
            s = conf_styles.get((conf_key, row_idx))
            return f' style="{s}"' if s else ""

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

        # Data rows with model-colored left border and landmark highlights
        for ri, row in enumerate(pivot_data):
            model_val = row.get("Model", "")
            border_color = MODEL_COLORS.get(model_val, "#E5E0DB")
            html += f'<tr style="border-left:4px solid {border_color}">'
            for h in factor_hdrs:
                html += f'<td class="factor">{row.get(h, "")}</td>'
            for acc_key, conf_key in [("Accuracy", "Confidence"), ("Acc Easy", "Conf Easy"), ("Acc Hard", "Conf Hard")]:
                acc_v = row[acc_key]
                conf_v = row[conf_key]
                acc_bg = _acc_highlight(acc_v, acc_key)
                conf_bg = _conf_highlight(conf_key, ri)
                html += f'<td{acc_bg}>{acc_v}</td><td{conf_bg}>{conf_v}</td>'
            html += f'<td>{row["N"]}</td>'
            html += "</tr>"

        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption(
            "Accuracy: **teal** = best per column (spread \u2265 1 pp), "
            "**rose** = worst (spread \u2265 3 pp). "
            "Confidence: **teal** = 3 best-calibrated per column (smallest |conf \u2212 acc|), "
            "**rose** = 3 most overconfident. Bold = rank 1."
        )

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
                    row[col] = f"{vals.mean():.2f}" if len(vals) > 0 else "-"
                summary_rows.append(row)
            _html_table(summary_rows, heatmap_cols=available)
            st.caption(
                "Blue intensity \u221d column-normalised signal strength (min\u2013max scaling)."
            )

    # Calibration curves — one chart per prompt mode
    st.subheader("Calibration: Reliability Diagram")
    st.caption("ECE = weighted mean |accuracy - confidence|. One chart per prompt mode.")
    _plot_calibration_by_mode(df)


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 1.0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _plot_calibration_by_mode(df: pd.DataFrame) -> None:
    """Render one reliability diagram per prompt mode (direct / cot / think).

    Filtering per mode to reduce clutter:
    - Direct: base (noshuffle, nopara) + shuffle+para
    - CoT / Think: base only (noshuffle, nopara)
    """
    df_valid = df[(df["num_queries"] > 0) & df["is_correct"].notna()].copy()
    if df_valid.empty:
        st.info("No data for calibration.")
        return

    modes_present = sorted(df_valid["mode"].unique()) if "mode" in df_valid.columns else ["direct"]
    for mode in modes_present:
        mode_df = df_valid[df_valid["mode"] == mode] if "mode" in df_valid.columns else df_valid
        if mode_df.empty:
            continue
        base_mask = (~mode_df["shuffle"]) & (~mode_df["paraphrase"])
        if mode == "direct":
            both_mask = mode_df["shuffle"] & mode_df["paraphrase"]
            mode_df = mode_df[base_mask | both_mask]
        else:
            mode_df = mode_df[base_mask]
        if mode_df.empty:
            continue
        _plot_calibration_single(mode_df, title=f"{mode.capitalize()} — Confidence vs Accuracy")


def _plot_calibration_single(df_valid: pd.DataFrame, title: str = "Calibration",
                             n_bins: int = 16, min_count: int = 2) -> None:
    """Binned reliability diagram for a single prompt-mode subset."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0.25, 1], y=[0.25, 1], mode="lines",
        line=dict(color="#000000", dash="dash", width=1.5),
        name="Perfect", showlegend=True,
    ))

    LINE_STYLES = ["solid", "dot", "dashdot", "longdash", "dash"]

    run_names = sorted(df_valid["run_name"].unique())
    model_run_count: dict[str, int] = {}
    degenerate_runs: list[str] = []
    for ri, run_name in enumerate(run_names):
        sub = df_valid[df_valid["run_name"] == run_name]
        if len(sub) < 5:
            continue

        cfg = _cfg_for_run_name(run_name)
        label = format_run_label(cfg) if cfg else run_name.replace("quality_", "")

        model_name = _short_model_name(cfg) if cfg else "Unknown"
        color = MODEL_COLORS.get(model_name, PLOT_COLORS[ri % len(PLOT_COLORS)])
        style_idx = model_run_count.get(model_name, 0)
        model_run_count[model_name] = style_idx + 1
        dash_style = LINE_STYLES[style_idx % len(LINE_STYLES)]

        msp_vals = sub["msp"].values
        correct_vals = sub["is_correct"].astype(float).values

        lo, hi = 0.25, 1.0
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        bin_centers, bin_accs, bin_counts = [], [], []

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

        ece = 0.0
        total_n = sum(bin_counts)
        for acc, conf, n in zip(bin_accs, bin_centers, bin_counts):
            if not math.isnan(acc) and total_n > 0:
                ece += (n / total_n) * abs(acc - conf)

        valid = [(c, a) for c, a in zip(bin_centers, bin_accs) if not math.isnan(a)]
        if valid:
            plot_x, plot_y = zip(*valid)
            suffix = " · MSP≈1" if len(valid) <= 2 else ""
            if len(valid) <= 2:
                degenerate_runs.append(label)
            fig.add_trace(go.Scatter(
                x=list(plot_x), y=list(plot_y),
                mode="lines+markers",
                line=dict(color=color, width=1.8, dash=dash_style),
                marker=dict(size=6 if len(valid) <= 2 else 4, color=color),
                name=f"{label} (ECE={ece:.2f}){suffix}",
            ))

    fig.update_layout(**_base_layout(
        title=title,
        xaxis_title="MSP (model confidence)",
        yaxis_title="Accuracy in bin",
        xaxis=dict(range=[0.2, 1.02], gridcolor=GRID),
        yaxis=dict(range=[0, 1.05], gridcolor=GRID),
        legend=dict(x=0.02, y=0.98),
        height=380,
    ))

    st.plotly_chart(_round_hover(fig), width="stretch")

    if degenerate_runs:
        degen_list = ", ".join(degenerate_runs)
        st.caption(f"\u26a0 {degen_list}: near-degenerate MSP (\u22481.0 for almost all questions) "
                   "\u2014 too little spread for a meaningful calibration curve. Shown as a single marker.")

    st.caption("Reliability diagram: each point is the actual accuracy of questions whose mean "
               "confidence (MSP) falls in that bin. A perfectly calibrated model follows the "
               "diagonal. ECE = weighted mean absolute gap between confidence and accuracy across bins.")


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

    # Build dropdown by unique question (not per-run)
    _has_qtext = "question_text" in df_valid.columns
    _id_cols = ["question_id", "difficult"] + (["question_text"] if _has_qtext else [])
    q_info = df_valid.drop_duplicates("question_id")[_id_cols].copy()
    if _has_qtext:
        q_info["label"] = q_info["question_id"].str[:20] + " | " + q_info["question_text"].str[:60]
    else:
        q_info["label"] = q_info["question_id"].str[:40]
    selected_label = st.selectbox("Select a question", q_info["label"].values)
    if not selected_label:
        return

    qid = q_info[q_info["label"] == selected_label]["question_id"].iloc[0]
    all_rows = df_valid[df_valid["question_id"] == qid]
    first = all_rows.iloc[0]

    # --- Question header (shared across runs) ---
    if _has_qtext:
        st.markdown(f"**Q: {first['question_text']}**")
    else:
        st.markdown(f"**Q: {qid}**")
    options = first.get("options", []) if "options" in first.index else []
    correct = first.get("correct_answer", -1)
    diff_label = "Hard" if first["difficult"] else "Easy"
    if options:
        opts_html = ""
        for i, opt in enumerate(options):
            marker = ' <span style="color:#2A8C8F; font-weight:600;">[CORRECT]</span>' if i == correct else ""
            opts_html += f'<div style="font-size:13px; margin:2px 0;"><b>{ANSWER_LETTERS[i]})</b> {opt}{marker}</div>'
        st.markdown(opts_html, unsafe_allow_html=True)
    st.caption(f"Difficulty: {diff_label} · Article: {first.get('article_id', '?')}")

    # --- Article context ---
    article_texts = load_article_texts()
    article_id = first.get("article_id", "")
    article_text = article_texts.get(article_id, "")
    if article_text:
        with st.expander(f"Article context (article_id: {article_id})", expanded=False):
            escaped = article_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            paragraphs = escaped.split("\n\n")
            html = "".join(f"<p>{' '.join(p.split())}</p>" for p in paragraphs)
            st.markdown(
                f'<div style="font-size:14px; line-height:1.6; max-height:500px; '
                f'overflow-y:auto; white-space:normal; word-wrap:break-word;">'
                f'{html}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # --- Per-run results ---
    for ri, (_, row) in enumerate(all_rows.iterrows()):
        # Build run label
        cfg = _cfg_for_run_name(row["run_name"])
        label = format_run_label(cfg) if cfg else row["run_name"].replace("quality_", "")
        prompt_mode = cfg.get("prompt_mode", "direct") if cfg else "direct"

        correct_str = "Correct" if row["is_correct"] else "Wrong"
        correct_color = TEAL if row["is_correct"] else ROSE
        agree_str = f'{row["agreement"]:.2f}' if not math.isnan(row.get("agreement", float("nan"))) else "-"
        epist_str = f'{row["epistemic"]:.2f}' if not math.isnan(row.get("epistemic", float("nan"))) else "-"

        st.markdown(
            f'<div style="font-size:14px; margin-bottom:4px;">'
            f'<b>{label}</b> · '
            f'Answer: <b>{ANSWER_LETTERS[row["final_answer"]]}</b> · '
            f'<span style="color:{correct_color};">{correct_str}</span> · '
            f'MSP: {row["msp"]:.2f} · '
            f'Agreement: {agree_str} · '
            f'Epistemic: {epist_str}'
            f'</div>',
            unsafe_allow_html=True,
        )

        query_log = row.get("query_log", [])

        # Per-query probability bar chart (only if multiple queries)
        if len(query_log) > 1:
            fig = go.Figure()
            for qi, ql in enumerate(query_log):
                probs = ql.get("canonical_probs", [0.25]*4)
                for li, letter in enumerate(ANSWER_LETTERS):
                    fig.add_trace(go.Bar(
                        x=[f"Q{qi}"], y=[probs[li] if li < len(probs) else 0],
                        name=letter,
                        marker_color=CHOICE_COLORS[li],
                        showlegend=(qi == 0), legendgroup=letter,
                    ))
            fig.update_layout(**_base_layout(
                xaxis_title="Query", yaxis_title="Probability",
                barmode="stack", height=220, showlegend=True,
            ))
            fig.update_traces(hovertemplate="%{fullData.name}: %{y:.2f}<extra></extra>")
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Stacked probability vectors per query (shuffle permutation). "
                "Consistent stacking = stable prediction; shifting colours = position-sensitive."
            )

        # Reasoning trace (CoT / think modes)
        if prompt_mode in ("cot", "cot_structured") or (cfg and cfg.get("think")):
            # Pass 1 vs Pass 2 answer — always visible
            pass1_canonical = query_log[0].get("pass1_canonical_answer", -1) if query_log else -1
            if pass1_canonical >= 0:
                pass1_letter = ANSWER_LETTERS[pass1_canonical]
                pass2_letter = ANSWER_LETTERS[row['final_answer']]
                agree = "agree" if pass1_letter == pass2_letter else "**DISAGREE**"
                st.caption(f"Pass 1 (free) answer: **{pass1_letter}** · Pass 2 (logprob) answer: **{pass2_letter}** ({agree})")

            # Reasoning trace in expander
            traces = [ql.get("thinking_trace", "") for ql in query_log if ql.get("thinking_trace")]
            if traces:
                with st.expander("Reasoning trace", expanded=False):
                    trace_text = traces[0]
                    escaped = trace_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    paras = escaped.split("\n\n")
                    html = "".join(f"<p>{' '.join(p.split())}</p>" for p in paras)
                    st.markdown(
                        f'<div style="font-size:13px; line-height:1.5; max-height:400px; '
                        f'overflow-y:auto; color:#4a5568;">{html}</div>',
                        unsafe_allow_html=True,
                    )

        if ri < len(all_rows) - 1:
            st.divider()


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

    # Index runs by condition tuple: (model, mode, shuffle, paraphrase)
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
            "epistemic": sub["epistemic"].mean() if "epistemic" in sub.columns and len(sub["epistemic"].dropna()) > 0 else None,
            "epistemic_correct": correct_sub["epistemic"].mean() if "epistemic" in correct_sub.columns and len(correct_sub["epistemic"].dropna()) > 0 else None,
            "epistemic_incorrect": incorrect_sub["epistemic"].mean() if "epistemic" in incorrect_sub.columns and len(incorrect_sub["epistemic"].dropna()) > 0 else None,
            "model": sub["model"].iloc[0] if "model" in sub.columns else "Unknown",
            "mode": sub["mode"].iloc[0] if "mode" in sub.columns else "direct",
            "shuffle": sub["shuffle"].iloc[0] if "shuffle" in sub.columns else True,
            "paraphrase": sub["paraphrase"].iloc[0] if "paraphrase" in sub.columns else False,
            "context": sub["context_condition"].iloc[0],
        }

    # --- Summary metrics table (like old dashboard) ---
    st.subheader("Summary Metrics by Run")
    summary_rows = []
    for run_name, m in sorted(run_metrics.items()):
        # Build label from config
        cfg = _cfg_for_run_name(run_name)
        label = format_run_label(cfg) if cfg else run_name.replace("quality_", "")

        def _delta_str(correct_val, incorrect_val):
            if correct_val is None or incorrect_val is None:
                return "-"
            return f"{incorrect_val - correct_val:+.2f}"

        summary_rows.append({
            "Model": m.get("model", "Unknown"),
            "Run": label,
            "N": m["n"],
            "Accuracy": f"{m['accuracy']:.1%}" if m["accuracy"] is not None else "-",
            "MSP (correct)": f"{m['msp_correct']:.2f}" if m["msp_correct"] is not None else "-",
            "MSP delta": _delta_str(m["msp_correct"], m["msp_incorrect"]),
            "Agree (correct)": f"{m['agreement_correct']:.2f}" if m["agreement_correct"] is not None else "-",
            "Agree delta": _delta_str(m["agreement_correct"], m["agreement_incorrect"]),
            "Epist. (correct)": f"{m['epistemic_correct']:.2f}" if m["epistemic_correct"] is not None else "-",
            "Epist. delta": _delta_str(m["epistemic_correct"], m["epistemic_incorrect"]),
        })

    if summary_rows:
        summary_rows.sort(key=lambda r: float(r["Accuracy"].rstrip("%")) if r["Accuracy"] != "-" else 0, reverse=True)
        _html_table(summary_rows, highlight={
            "MSP delta": "neg_good",
            "Agree delta": "neg_good",
            "Epist. delta": "pos_good",
        })
        st.caption(
            "Colour intensity \u221d effect size (normalised to column max). "
            "**Teal** = delta in the direction expected for a useful signal, "
            "**rose** = opposite direction."
        )

    # --- Matched pair analysis ---
    st.subheader("Matched Pair Effects")

    # Build pairwise comparisons: each compares two values of one variable,
    # matched on all other variables. Mode comparisons are pairwise against direct.
    comparisons = []  # (var_name, val_a, val_b, label)

    shuffles = set(m["shuffle"] for m in run_metrics.values())
    modes = set(m["mode"] for m in run_metrics.values())
    paraphrases = set(m["paraphrase"] for m in run_metrics.values())

    if len(shuffles) > 1:
        comparisons.append(("shuffle", True, False, "Shuffle on vs off"))
    if len(paraphrases) > 1:
        comparisons.append(("paraphrase", True, False, "Paraphrase on vs off"))
    if "CoT" in modes and "direct" in modes:
        comparisons.append(("mode", "CoT", "direct", "CoT vs direct"))
    if "think" in modes and "direct" in modes:
        comparisons.append(("mode", "think", "direct", "Think vs direct"))
    if "think" in modes and "CoT" in modes:
        comparisons.append(("mode", "think", "CoT", "Think vs CoT"))

    if not comparisons:
        st.info("Need runs with contrasting conditions to compute effects.")
        return

    match_vars = ["model", "context", "shuffle", "mode", "paraphrase"]
    effect_rows = []
    run_list = list(run_metrics.items())

    for var_name, val_a, val_b, label in comparisons:
        acc_deltas = []
        msp_deltas = []
        epist_deltas = []
        pairs_found = 0

        for i, (name_a, m_a) in enumerate(run_list):
            for name_b, m_b in run_list[i+1:]:
                # One must have val_a, the other val_b
                a_has_a = m_a[var_name] == val_a and m_b[var_name] == val_b
                b_has_a = m_b[var_name] == val_a and m_a[var_name] == val_b
                if not (a_has_a or b_has_a):
                    continue
                # Must match on all other variables
                other = [v for v in match_vars if v != var_name]
                if not all(m_a[v] == m_b[v] for v in other):
                    continue
                pairs_found += 1
                # Delta = val_a - val_b
                if a_has_a:
                    ma, mb = m_a, m_b
                else:
                    ma, mb = m_b, m_a
                if ma["accuracy"] is not None and mb["accuracy"] is not None:
                    acc_deltas.append(ma["accuracy"] - mb["accuracy"])
                if ma["msp_correct"] is not None and mb["msp_correct"] is not None:
                    msp_deltas.append(ma["msp_correct"] - mb["msp_correct"])
                if ma["epistemic"] is not None and mb["epistemic"] is not None:
                    epist_deltas.append(ma["epistemic"] - mb["epistemic"])

        avg_acc = sum(acc_deltas) / len(acc_deltas) if acc_deltas else None
        avg_msp = sum(msp_deltas) / len(msp_deltas) if msp_deltas else None
        avg_epist = sum(epist_deltas) / len(epist_deltas) if epist_deltas else None
        effect_rows.append({
            "Effect": label,
            "Pairs": pairs_found,
            "Acc. Delta": f"{avg_acc:+.1%}" if avg_acc is not None else "-",
            "MSP Delta": f"{avg_msp:+.2f}" if avg_msp is not None else "-",
            "Epistemic Delta": f"{avg_epist:+.2f}" if avg_epist is not None else "-",
        })

    if effect_rows:
        _html_table(effect_rows, highlight={
            "Acc. Delta": "pos_good",
            "MSP Delta": "pos_good",
            "Epistemic Delta": "pos_good",
        })
        st.caption(
            "Colour intensity \u221d effect size (normalised to column max). "
            "**Teal** = positive effect (condition A improves metric), "
            "**rose** = negative effect."
        )

    # --- Per-run distribution comparison ---
    st.subheader("MSP Distributions by Run")
    st.caption("Compare how confidence distributions shift across conditions.")

    fig = go.Figure()
    for ri, run_name in enumerate(sorted(active["run_name"].unique())):
        sub = active[(active["run_name"] == run_name) & active["msp"].notna()]
        if len(sub) == 0:
            continue

        cfg = _cfg_for_run_name(run_name)
        label = format_run_label(cfg) if cfg else run_name.replace("quality_", "")
        # Color by model for visual grouping
        model_name = _short_model_name(cfg) if cfg else "Unknown"
        color = MODEL_COLORS.get(model_name, PLOT_COLORS[ri % len(PLOT_COLORS)])

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
    st.plotly_chart(_round_hover(fig), width="stretch")


# ---------------------------------------------------------------------------
# Tab 6: Signal Battery
# ---------------------------------------------------------------------------

ANSWER_TOKENS = {" A", " B", " C", " D", "A", "B", "C", "D"}


def _html_table(rows: list[dict], highlight: dict[str, str] | None = None,
                heatmap_cols: list[str] | None = None) -> None:
    """Render a list of dicts as a styled HTML table.

    Args:
        rows: List of dicts, one per row.
        highlight: Optional dict of column_name -> "neg_good" or "pos_good" for
            directional coloring (teal for good, rose for bad).
        heatmap_cols: Optional list of columns to apply neutral blue heatmap
            (column-wise min-max scaling). Useful for signal tables.
    """
    if not rows:
        return
    highlight = highlight or {}
    heatmap_cols = heatmap_cols or []
    cols = list(rows[0].keys())

    # Pre-compute per-column min/max for heatmap
    col_range: dict[str, tuple[float, float]] = {}
    for c in heatmap_cols:
        vals = []
        for r in rows:
            v = r.get(c, "-")
            if isinstance(v, str) and v not in ("-", ""):
                try:
                    vals.append(float(v.replace("%", "").replace("+", "")))
                except ValueError:
                    pass
        if vals:
            col_range[c] = (min(vals), max(vals))

    # Collect max absolute value across highlight columns
    max_abs = 0.0
    if highlight:
        for r in rows:
            for c in highlight:
                val = r.get(c, "-")
                if isinstance(val, str) and val not in ("-", ""):
                    try:
                        max_abs = max(max_abs, abs(float(val.replace("%", "").replace("+", ""))))
                    except ValueError:
                        pass
    if max_abs == 0:
        max_abs = 1.0

    html = """<style>
    .sig-table { border-collapse: collapse; width: 100%; font-family: Inter, sans-serif; font-size: 13px; }
    .sig-table th, .sig-table td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #E5E0DB; }
    .sig-table th { color: #8B95A1; font-weight: 500; }
    .sig-table td:first-child { text-align: left; font-weight: 500; }
    .sig-table tr:hover { background: #F5F3F1; }
    </style><table class="sig-table"><tr>"""
    for c in cols:
        html += f"<th>{c}</th>"
    html += "</tr>"
    for r in rows:
        html += "<tr>"
        for c in cols:
            val = r.get(c, "-")
            bg = ""
            # Directional highlight (for effect-size tables)
            if c in highlight and isinstance(val, str) and val not in ("-", ""):
                try:
                    fval = float(val.replace("%", "").replace("+", ""))
                    direction = highlight[c]
                    good = (fval < 0 and direction == "neg_good") or (fval > 0 and direction == "pos_good")
                    strength = min(abs(fval) / max_abs, 1.0)
                    alpha = strength ** 2 * 0.35
                    if good and alpha > 0.03:
                        bg = f' style="background:rgba(42,140,143,{alpha:.2f});"'
                    elif not good and alpha > 0.03:
                        bg = f' style="background:rgba(220,100,100,{alpha:.2f});"'
                except ValueError:
                    pass
            # Neutral heatmap (for signal overview tables)
            elif c in col_range and isinstance(val, str) and val not in ("-", ""):
                try:
                    fval = float(val.replace("%", "").replace("+", ""))
                    cmin, cmax = col_range[c]
                    span = cmax - cmin
                    if span > 0:
                        t = (fval - cmin) / span
                        alpha = 0.06 + t * 0.24
                        bg = f' style="background:rgba(42,100,160,{alpha:.2f});"'
                except ValueError:
                    pass
            html += f"<td{bg}>{val}</td>"
        html += "</tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


def _compute_signal_battery(df: pd.DataFrame) -> list[dict]:
    """Compute per-run signal deltas (incorrect - correct) from the main dataframe."""
    rows = []
    for run_name in df["run_name"].unique():
        sub = df[df["run_name"] == run_name]
        valid = sub[sub["is_correct"].notna()]
        if len(valid) == 0:
            continue

        correct_sub = valid[valid["is_correct"] == True]
        incorrect_sub = valid[valid["is_correct"] == False]
        cfg = _cfg_for_run_name(run_name)

        def _delta(correct_vals: list, incorrect_vals: list) -> float | None:
            """Compute incorrect_mean - correct_mean. Positive = signal is higher for wrong answers."""
            if not correct_vals or not incorrect_vals:
                return None
            return sum(incorrect_vals) / len(incorrect_vals) - sum(correct_vals) / len(correct_vals)

        # --- MSP delta ---
        msp_d = _delta(
            correct_sub["msp"].tolist(),
            incorrect_sub["msp"].tolist(),
        )

        # --- Epistemic delta ---
        epist_d = None
        if "epistemic" in valid.columns:
            c_ep = correct_sub["epistemic"].dropna().tolist()
            i_ep = incorrect_sub["epistemic"].dropna().tolist()
            epist_d = _delta(c_ep, i_ep)

        # --- S3: Second-choice gap ---
        gap_c, gap_i = [], []
        for _, r in valid.iterrows():
            mp = sorted(r["mean_probs"], reverse=True)
            if len(mp) >= 2:
                g = mp[0] - mp[1]
                (gap_c if r["is_correct"] else gap_i).append(g)
        gap_d = _delta(gap_c, gap_i)

        # --- S5: Answer coverage ---
        cov_c, cov_i = [], []
        for _, r in valid.iterrows():
            ql = r.get("query_log", [])
            q_covs = []
            for q in ql:
                top_lps = []
                for entry in q.get("raw_logprobs", []):
                    for tlp in entry.get("top_logprobs", []):
                        top_lps.append((tlp.get("token", ""), tlp.get("logprob", -30)))
                if top_lps:
                    total = sum(math.exp(lp) for _, lp in top_lps)
                    ans = sum(math.exp(lp) for tok, lp in top_lps if tok.strip() in ANSWER_TOKENS)
                    if total > 0:
                        q_covs.append(ans / total)
            if q_covs:
                (cov_c if r["is_correct"] else cov_i).append(sum(q_covs) / len(q_covs))
        cov_d = _delta(cov_c, cov_i)

        # --- S14: Confidence variance ---
        cv_c, cv_i = [], []
        for _, r in valid.iterrows():
            ql = r.get("query_log", [])
            pqc = [max(q.get("canonical_probs", [0.25]*4)) for q in ql
                   if len(q.get("canonical_probs", [])) == 4]
            if len(pqc) > 1:
                v = float(np.std(pqc))
                (cv_c if r["is_correct"] else cv_i).append(v)
        cv_d = _delta(cv_c, cv_i)

        # --- S15c: Agreement-Confidence gap ---
        ac_c, ac_i = [], []
        for _, r in valid.iterrows():
            if not math.isnan(r.get("agreement", float("nan"))):
                g = r["agreement"] - r["msp"]
                (ac_c if r["is_correct"] else ac_i).append(g)
        ac_d = _delta(ac_c, ac_i)

        model_name = _short_model_name(cfg) if cfg else "Unknown"
        mode = _effective_mode(cfg) if cfg else "direct"
        shuffle = cfg.get("shuffle_options", False) if cfg else False
        para = cfg.get("use_paraphrases", False) if cfg else False

        rows.append({
            "Model": model_name,
            "Mode": mode,
            "Shuffle": "Yes" if shuffle else "No",
            "Para": "Yes" if para else "No",
            "N": len(valid),
            "MSP": msp_d,
            "Epistemic": epist_d,
            "2nd Gap": gap_d,
            "Coverage": cov_d,
            "Conf Var": cv_d,
            "Agree-Conf": ac_d,
        })

    return rows


def tab_signals() -> None:
    st.header("Signal Battery")
    st.caption(
        "Delta = mean(incorrect) - mean(correct). Negative MSP/Gap/Coverage = signal is lower when wrong (good). "
        "Positive Epistemic/Conf Var = signal is higher when wrong (good). Stronger colour = more discriminative."
    )

    if df.empty:
        st.info("No data loaded.")
        return

    rows = _compute_signal_battery(df)
    if not rows:
        st.info("No data to compute signals.")
        return

    rows.sort(key=lambda r: r["N"], reverse=True)

    single_cols = ["MSP", "2nd Gap", "Coverage"]
    multi_cols = ["Epistemic", "Conf Var", "Agree-Conf"]

    def _signal_table(title: str, caption: str, cols: list[str], neg_good: set[str], row_override: list[dict] | None = None) -> None:
        """Render one signal table with colour highlighting."""
        st.subheader(title)
        st.caption(caption)
        table_rows = row_override if row_override is not None else rows

        all_abs = []
        for r in table_rows:
            for col in cols:
                v = r.get(col)
                if v is not None:
                    all_abs.append(abs(v))
        max_abs = max(all_abs) if all_abs else 1.0

        html = """<style>
        .sig-table { border-collapse: collapse; width: 100%; font-family: Inter, sans-serif; font-size: 13px; }
        .sig-table th, .sig-table td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #E5E0DB; }
        .sig-table th { color: #8B95A1; font-weight: 500; }
        .sig-table td.cond { text-align: left; font-weight: 500; }
        .sig-table tr:hover { background: #F5F3F1; }
        </style><table class="sig-table">"""

        html += "<tr><th>Model</th><th>Mode</th><th>Shuffle</th><th>Para</th><th>N</th>"
        for col in cols:
            html += f"<th>{col}</th>"
        html += "</tr>"

        for r in table_rows:
            html += "<tr>"
            html += f'<td class="cond">{r.get("Model", "")}</td><td class="cond">{r["Mode"]}</td><td>{r["Shuffle"]}</td><td>{r["Para"]}</td><td>{r["N"]}</td>'
            for col in cols:
                v = r.get(col)
                if v is None:
                    html += "<td>-</td>"
                    continue
                strength = (-v if col in neg_good else v) / max_abs
                strength = max(-1.0, min(1.0, strength))
                if strength > 0.05:
                    alpha = strength ** 2 * 0.35
                    bg = f"rgba(42, 140, 143, {alpha:.2f})"
                elif strength < -0.05:
                    alpha = strength ** 2 * 0.35
                    bg = f"rgba(220, 100, 100, {alpha:.2f})"
                else:
                    bg = "transparent"
                html += f'<td style="background:{bg};">{v:+.2f}</td>'
            html += "</tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption(
            "Colour intensity \u221d signal strength\u00b2 (normalised to column max). "
            "**Teal** = signal in the expected direction for detecting errors, "
            "**rose** = opposite direction."
        )

    _signal_table(
        "Single-Query Signals",
        "From individual forward passes. Delta = mean(incorrect) - mean(correct). "
        "Negative = signal is lower when wrong (good for MSP, 2nd Gap, Coverage).",
        single_cols, neg_good={"MSP", "2nd Gap", "Coverage"},
    )

    st.markdown("""
    - **MSP**: max probability. Lower when wrong = less confident.
    - **2nd Gap**: `top_prob - second_prob` on mean distribution. Lower when wrong = less decisive.
    - **Coverage**: answer-token mass in top-20 logprobs. Lower when wrong = model hesitates. (Direct mode only.)
    """)

    multi_rows = [r for r in rows if r.get("Shuffle") == "Yes" or r.get("Para") == "Yes"]
    _signal_table(
        "Multi-Query Signals",
        "From comparing across queries (shuffle/paraphrase). Delta = mean(incorrect) - mean(correct). "
        "Positive = signal is higher when wrong (good for Epistemic, Conf Var, Agree-Conf).",
        multi_cols, neg_good=set(), row_override=multi_rows,
    )

    st.markdown("""
    - **Epistemic**: mutual information across queries. Higher when wrong = more disagreement.
    - **Conf Var**: std of per-query confidence. Higher when wrong = unstable confidence.
    - **Agree-Conf**: `agreement - MSP`. Higher when wrong = agrees on answer but not confident.
    """)

    # --- Cross-mode disagreement ---
    st.subheader("Cross-Mode Disagreement")
    st.caption(
        "When direct and CoT/think give different answers for the same question, "
        "how often is the disagreeing question wrong? Computed by joining results across conditions."
    )

    # Build per-question answer lookup for each mode (noshuffle only for fair comparison)
    mode_answers: dict[str, dict[str, int]] = {}  # mode -> {qid: final_answer}
    mode_correct: dict[str, dict[str, bool]] = {}
    for run_name in df["run_name"].unique():
        sub = df[df["run_name"] == run_name]
        if len(sub) == 0:
            continue
        cfg_r = _cfg_for_run_name(run_name)
        if not cfg_r:
            continue
        mode = _effective_mode(cfg_r)
        shuffle = cfg_r.get("shuffle_options", False)
        para = cfg_r.get("use_paraphrases", False)
        # Use noshuffle, no-para runs for clean comparison
        if shuffle or para:
            continue
        answers = {}
        corrects = {}
        for _, r in sub.iterrows():
            qid = r["question_id"]
            answers[qid] = r["final_answer"]
            corrects[qid] = r["is_correct"]
        mode_answers[mode] = answers
        mode_correct[mode] = corrects

    cross_pairs = [
        ("CoT", "direct"), ("think", "direct"), ("think", "CoT"),
    ]
    cross_rows = []
    for mode_a, mode_b in cross_pairs:
        if mode_a not in mode_answers or mode_b not in mode_answers:
            continue
        ans_a = mode_answers[mode_a]
        ans_b = mode_answers[mode_b]
        common = set(ans_a.keys()) & set(ans_b.keys())
        if not common:
            continue
        agree_count = 0
        disagree_count = 0
        agree_acc = []
        disagree_acc = []
        for qid in common:
            correct = mode_correct[mode_a].get(qid)  # use mode_a's correctness
            if ans_a[qid] == ans_b[qid]:
                agree_count += 1
                if correct is not None:
                    agree_acc.append(correct)
            else:
                disagree_count += 1
                if correct is not None:
                    disagree_acc.append(correct)
        total = agree_count + disagree_count
        cross_rows.append({
            "Comparison": f"{mode_a} vs {mode_b}",
            "N": total,
            "Agree": f"{agree_count} ({agree_count/total:.0%})" if total else "-",
            "Disagree": f"{disagree_count} ({disagree_count/total:.0%})" if total else "-",
            "Acc (agree)": f"{sum(agree_acc)/len(agree_acc):.1%}" if agree_acc else "-",
            "Acc (disagree)": f"{sum(disagree_acc)/len(disagree_acc):.1%}" if disagree_acc else "-",
        })

    if cross_rows:
        _html_table(cross_rows)
        st.markdown("When modes **agree**, accuracy is high. When they **disagree**, accuracy drops — the disagreement itself is a strong uncertainty signal.")
    else:
        st.info("Need at least 2 modes (direct, CoT, think) with base runs (no shuffle/paraphrase) to compare.")

    # --- Position loyalty (shuffle runs only) ---
    st.subheader("Position Loyalty")
    st.caption(
        "How much does the correct answer's probability change based on its display position? "
        "High variance = model relies on position, not content. (Shuffle runs only.)"
    )

    pos_rows = []
    for run_name in df["run_name"].unique():
        sub = df[df["run_name"] == run_name]
        if len(sub) == 0:
            continue
        cfg_r = _cfg_for_run_name(run_name)
        if not cfg_r or not cfg_r.get("shuffle_options", False):
            continue

        loyalty_correct = []
        loyalty_incorrect = []
        for _, r in sub[sub["is_correct"].notna()].iterrows():
            ql = r.get("query_log", [])
            if len(ql) < 2:
                continue
            correct_ans = r["correct_answer"]
            # Get the correct answer's probability across different permutations
            correct_probs = []
            for q in ql:
                probs = q.get("canonical_probs", [])
                if len(probs) > correct_ans:
                    correct_probs.append(probs[correct_ans])
            if len(correct_probs) > 1:
                var = float(np.std(correct_probs))
                (loyalty_correct if r["is_correct"] else loyalty_incorrect).append(var)

        mode = _effective_mode(cfg_r)
        para = cfg_r.get("use_paraphrases", False)
        mean_c = sum(loyalty_correct) / len(loyalty_correct) if loyalty_correct else None
        mean_i = sum(loyalty_incorrect) / len(loyalty_incorrect) if loyalty_incorrect else None
        delta = (mean_i - mean_c) if mean_c is not None and mean_i is not None else None

        pos_rows.append({
            "Mode": mode,
            "Para": "Yes" if para else "No",
            "N": len(loyalty_correct) + len(loyalty_incorrect),
            "Pos Var (correct)": f"{mean_c:.2f}" if mean_c is not None else "-",
            "Pos Var (incorrect)": f"{mean_i:.2f}" if mean_i is not None else "-",
            "Delta": f"{delta:+.2f}" if delta is not None else "-",
        })

    if pos_rows:
        _html_table(pos_rows, highlight={"Delta": "pos_good"})
        st.caption(
            "Delta colour: **teal** = higher variance on incorrect (expected), "
            "**rose** = opposite. Intensity \u221d effect size."
        )
        st.markdown("Higher position variance on incorrect answers means the model's wrong answers are more position-dependent — relying on heuristics rather than comprehension.")
    else:
        st.info("Need shuffle runs to compute position loyalty.")

    # --- Reasoning trace length (CoT/think only) ---
    st.subheader("Reasoning Signals")
    st.caption(
        "Trace length + Pass 1/2 disagreement rate for CoT and think modes. "
        "Pass 1 = model's free answer, Pass 2 = logprob extraction. Higher disagreement on incorrect = signal works."
    )

    trace_rows = []
    for run_name in df["run_name"].unique():
        sub = df[df["run_name"] == run_name]
        if len(sub) == 0:
            continue
        cfg_r = _cfg_for_run_name(run_name)
        if not cfg_r:
            continue
        mode = _effective_mode(cfg_r)
        if mode == "direct":
            continue

        len_correct = []
        len_incorrect = []
        p1p2_disagree_correct = []
        p1p2_disagree_incorrect = []

        for _, r in sub[sub["is_correct"].notna()].iterrows():
            ql = r.get("query_log", [])

            # Trace length
            traces = [len(q.get("thinking_trace", "")) for q in ql if q.get("thinking_trace")]
            if traces:
                mean_len = sum(traces) / len(traces)
                (len_correct if r["is_correct"] else len_incorrect).append(mean_len)

            # Pass 1/2 disagreement rate per question
            disagree_count = 0
            total_count = 0
            for q in ql:
                p1c = q.get("pass1_canonical_answer", -1)
                p2c = q.get("canonical_answer", -1)
                if p1c >= 0:
                    total_count += 1
                    if p1c != p2c:
                        disagree_count += 1
            if total_count > 0:
                rate = disagree_count / total_count
                (p1p2_disagree_correct if r["is_correct"] else p1p2_disagree_incorrect).append(rate)

        shuffle = cfg_r.get("shuffle_options", False)
        para = cfg_r.get("use_paraphrases", False)
        mean_c = sum(len_correct) / len(len_correct) if len_correct else None
        mean_i = sum(len_incorrect) / len(len_incorrect) if len_incorrect else None
        trace_delta = (mean_i - mean_c) if mean_c is not None and mean_i is not None else None

        p1p2_c = sum(p1p2_disagree_correct) / len(p1p2_disagree_correct) if p1p2_disagree_correct else None
        p1p2_i = sum(p1p2_disagree_incorrect) / len(p1p2_disagree_incorrect) if p1p2_disagree_incorrect else None

        trace_rows.append({
            "Mode": mode,
            "Shuffle": "Yes" if shuffle else "No",
            "N": len(len_correct) + len(len_incorrect),
            "Trace (correct)": f"{mean_c:.0f}" if mean_c is not None else "-",
            "Trace (incorrect)": f"{mean_i:.0f}" if mean_i is not None else "-",
            "Trace Delta": f"{trace_delta:+.0f}" if trace_delta is not None else "-",
            "P1/P2 disagree (correct)": f"{p1p2_c:.1%}" if p1p2_c is not None else "-",
            "P1/P2 disagree (incorrect)": f"{p1p2_i:.1%}" if p1p2_i is not None else "-",
        })

    if trace_rows:
        _html_table(trace_rows)
        st.markdown(
            "**Trace Delta**: positive = model writes more when wrong. "
            "**P1/P2 disagree**: rate at which the model's free answer differs from logprob answer. "
            "Higher on incorrect = the model's commitment and its probability distribution conflict more when it's wrong."
        )
    else:
        st.info("Need CoT or think runs to analyse reasoning signals.")


# ---------------------------------------------------------------------------
# Tab 7: Adaptive Sampling
# ---------------------------------------------------------------------------

def tab_adaptive() -> None:
    """Bayesian adaptive sampling with posterior stopping criterion."""
    import pickle as _pickle

    st.header("Adaptive Sampling")
    st.caption(
        "Five posterior aggregation methods compared side by side. "
        "Raw results (T=1) on top, temperature-calibrated results below. "
        "All methods stop when confidence > \u03c4. Capped questions escalate to CoT/think."
    )

    if df.empty:
        st.info("No data loaded.")
        return

    # --- Constants ---
    METHODS = [("Product", "prod"), ("Sum", "sum"), ("Dirichlet MLE", "mle"),
               ("MoM", "mom"), ("MoM + Bayes", "mom_bayes")]
    MCOLORS = {"prod": TEAL, "sum": ROSE, "mle": GOLD,
               "mom": SLATE, "mom_bayes": PURPLE}
    MKEY_TO_METHOD = {"prod": "product", "sum": "sum", "mle": "mle",
                      "mom": "mom", "mom_bayes": "mom_bayes"}
    tau_values = [0.00, 0.60, 0.70, 0.80, 0.90, 0.95]
    tau_values_cal = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

    # --- Selectors ---
    n_max = st.select_slider("N_max", options=list(range(2, 11)), value=10)

    # --- Try to load precomputed cache (cached to avoid re-reading 7MB on every refresh) ---
    _cache_path = Path(__file__).resolve().parent / "adaptive_cache.pkl"

    @st.cache_data(ttl=300)
    def _load_adaptive_cache(_mtime: float) -> dict | None:
        try:
            with open(_cache_path, "rb") as _f:
                return _pickle.load(_f)
        except Exception:
            return None

    cache = None
    if _cache_path.exists():
        _mtime = _cache_path.stat().st_mtime
        cache = _load_adaptive_cache(_mtime)
        if cache and cache.get("n_max") != n_max:
            cache = None
    if cache is None and analysis_cache is not None and analysis_cache.get("adaptive"):
        cache = analysis_cache["adaptive"]
        if cache and cache.get("n_max") != n_max:
            cache = None

    if cache is not None:
        model_data = cache["model_data"]
        model_names = cache["model_names"]
        cal_grids_cached = cache.get("cal_grids", {})
        import datetime as _dt
        _age = time.time() - cache.get("computed_at", 0)
        _age_str = f"{_age / 60:.0f} min ago" if _age < 3600 else f"{_age / 3600:.1f} h ago"
        st.caption(f"Using precomputed cache ({_age_str}). "
                   f"Run `python dashboard/precompute_adaptive.py` to refresh.")
    else:
        if not HAS_LOCAL_FILES:
            st.info("Adaptive sampling data not available in the analysis cache. "
                    "Rebuild the cache locally with the adaptive cache present.")
            return
        cal_grids_cached = {}

        # --- Discover runs ---
        run_meta: dict[str, dict] = {}
        for prefix, cfg in _run_name_cfg.items():
            if cfg:
                run_meta[prefix] = {
                    "label": format_run_label(cfg),
                    "model": _short_model_name(cfg),
                    "mode": _effective_mode(cfg),
                    "shuffle": cfg.get("shuffle_options", False),
                    "paraphrase": cfg.get("use_paraphrases", False),
                    "context": cfg.get("context_condition", "unknown"),
                }

        # --- Auto-discover per-model configs ---
        models_found: dict[str, dict] = {}
        for model_name in sorted({v["model"] for v in run_meta.values()}):
            model_runs = {k: v for k, v in run_meta.items() if v["model"] == model_name}
            base = next(
                (k for k, v in model_runs.items()
                 if v["mode"] == "direct" and v["shuffle"] and not v["paraphrase"]),
                None,
            )
            if not base:
                continue
            esc1 = next(
                (k for k, v in model_runs.items()
                 if v["mode"] == "CoT" and not v["shuffle"] and not v["paraphrase"]),
                None,
            )
            esc2 = next(
                (k for k, v in model_runs.items()
                 if v["mode"] == "think" and not v["shuffle"] and not v["paraphrase"]),
                None,
            )
            models_found[model_name] = {"base": base, "esc1": esc1, "esc2": esc2}

        if not models_found:
            st.info("Need at least one model with a shuffle direct run for adaptive sampling.")
            return

        # --- Build per-model question data ---
        def _answer_map(rn):
            if not rn:
                return {}
            s = df[df["run_name"] == rn]
            return dict(zip(s["question_id"], s["final_answer"]))

        def _probs_map(rn):
            if not rn:
                return {}
            s = df[df["run_name"] == rn]
            out = {}
            for _, row in s.iterrows():
                ql = row.get("query_log", [])
                for q in ql:
                    cp = q.get("canonical_probs", [])
                    if len(cp) == 4:
                        out[row["question_id"]] = cp
                        break
            return out

        model_data: dict[str, dict] = {}
        for model_name, cfg in models_found.items():
            base_df = df[df["run_name"] == cfg["base"]]
            if base_df.empty:
                continue
            esc1_map = _answer_map(cfg["esc1"])
            esc2_map = _answer_map(cfg["esc2"])
            esc1_probs = _probs_map(cfg["esc1"])
            esc2_probs = _probs_map(cfg["esc2"])
            questions = []
            for _, row in base_df.iterrows():
                qlog = row.get("query_log", [])
                probs = [q["canonical_probs"] for q in qlog
                         if q.get("canonical_probs") and len(q["canonical_probs"]) == 4]
                if len(probs) < 2:
                    continue
                qid = row["question_id"]
                questions.append({
                    "correct": row["correct_answer"],
                    "difficult": row.get("difficult", False),
                    "probs": probs[:n_max],
                    "esc1": esc1_map.get(qid),
                    "esc2": esc2_map.get(qid),
                    "esc1_probs": esc1_probs.get(qid),
                    "esc2_probs": esc2_probs.get(qid),
                })
            if not questions:
                continue
            n_q = len(questions)
            baseline_correct = sum(
                1 for q in questions
                if int(np.argmax(np.mean(q["probs"], axis=0))) == q["correct"]
            )
            model_data[model_name] = {
                "questions": questions, "n_q": n_q,
                "baseline_acc": baseline_correct / n_q,
                "has_esc1": cfg["esc1"] is not None,
                "has_esc2": cfg["esc2"] is not None,
                "esc1_run": cfg["esc1"], "esc2_run": cfg["esc2"],
            }

        if not model_data:
            st.info("No models with enough data for adaptive sampling.")
            return

        # --- Helpers (only needed for live computation) ---
        def _trigamma(x):
            x = np.asarray(x, dtype=np.float64)
            result = np.zeros_like(x)
            for _ in range(7):
                small = x < 7.0
                result += np.where(small, 1.0 / (x * x), 0.0)
                x = np.where(small, x + 1.0, x)
            ix = 1.0 / x
            ix2 = ix * ix
            result += ix + ix2 * (0.5 + ix * (1.0 / 6.0 + ix2 * (-1.0 / 30.0 + ix2 * (1.0 / 42.0))))
            return result

        def _inverse_digamma(y, max_iter=8):
            psi_1 = float(_digamma(1.0))
            denom = y - psi_1
            denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
            x = np.where(y >= -2.22, np.exp(y) + 0.5, -1.0 / denom)
            x = np.maximum(x, 1e-8)
            for _ in range(max_iter):
                dx = (_digamma(x) - y) / _trigamma(x)
                x = np.maximum(x - dx, 1e-8)
            return x

        def _exceedance_copula(alpha):
            alpha = np.asarray(alpha, dtype=np.float64)
            alpha = np.maximum(alpha, 1e-12)
            squeeze = alpha.ndim == 1
            if squeeze:
                alpha = alpha[np.newaxis, :]
            K = alpha.shape[-1]
            M = alpha.shape[0]

            if K < 3:
                a = alpha.max(axis=-1)
                b = alpha.min(axis=-1)
                out = 1.0 - _betainc(a, b, 0.5)
                return float(out[0]) if squeeze else out

            leader_idx = np.argmax(alpha, axis=-1)
            row_idx = np.arange(M)
            alpha_lead = alpha[row_idx, leader_idx]

            col_idx = np.arange(K)
            comp_mask = col_idx[None, :] != leader_idx[:, None]
            alpha_comp = alpha[comp_mask].reshape(M, K - 1)

            p_j = np.clip(
                1.0 - _betainc(alpha_lead[:, None], alpha_comp, 0.5),
                1e-15, 1.0 - 1e-15,
            )
            a_j = _norm.ppf(p_j)
            phi_j = _norm.pdf(a_j)

            S = alpha.sum(axis=-1)[:, None]
            al = alpha_lead[:, None]
            denom = np.maximum(S * S * (S + 1.0), 1e-300)
            var_j = (al * (S - al) + alpha_comp * (S - alpha_comp) + 2.0 * al * alpha_comp) / denom
            var_j = np.maximum(var_j, 1e-300)

            aj_i, ak_i = alpha_comp[:, :, None], alpha_comp[:, None, :]
            S3 = S[:, :, None]
            al3 = al[:, :, None]
            denom3 = np.maximum(S3 * S3 * (S3 + 1.0), 1e-300)
            cov_jk = (al3 * (S3 + aj_i + ak_i - al3) - aj_i * ak_i) / denom3
            rho_jk = cov_jk / np.sqrt(var_j[:, :, None] * var_j[:, None, :])

            log_p = np.log(p_j)
            log_prod_all = log_p.sum(axis=-1)
            prod_all = np.exp(log_prod_all)

            log_rem = log_prod_all[:, None, None] - log_p[:, :, None] - log_p[:, None, :]
            term = rho_jk * (phi_j[:, :, None] * phi_j[:, None, :]) * np.exp(log_rem)

            triu = np.triu(np.ones((K - 1, K - 1), dtype=bool), k=1)
            correction = term[:, triu].sum(axis=-1)

            damping = 0.637 + 0.206 * math.exp(-0.587 * (K - 3))
            P = np.clip(prod_all + damping * correction, 0.0, 1.0)
            return float(P[0]) if squeeze else P

        def _mom_estimate_alpha(mu, var_sum, mu_var_sum, N, K=4):
            if mu_var_sum < 1e-15:
                return mu * 10.0
            R = var_sum / mu_var_sum
            R = np.clip(R, 1e-6, 1.0 - 1e-6)
            alpha0 = (1.0 - R) / R
            alpha0 = np.clip(alpha0, 1.0, 200.0)
            alpha = alpha0 * mu
            return np.maximum(alpha, 1e-8)

        _BAYES_GRID = np.logspace(np.log10(0.5), np.log10(300), 80)

        def _bayesian_mom_exceedance(mu, var_sum, mu_var_sum, N, K=4):
            from scipy.stats import chi2 as _chi2, gamma as _gamma
            df_val = K * (N - 1)
            if mu_var_sum < 1e-15:
                return _exceedance_copula(mu * 10.0)

            R_hat = var_sum / mu_var_sum
            grid = _BAYES_GRID
            R_true = 1.0 / (grid + 1.0)
            ratio = R_hat / np.maximum(R_true, 1e-15) * df_val
            log_lik = _chi2.logpdf(ratio, df_val) + np.log(df_val) - np.log(np.maximum(R_true, 1e-15))
            log_prior = _gamma.logpdf(grid, a=2.0, scale=5.0)
            log_post = log_lik + log_prior
            log_post -= log_post.max()
            post = np.exp(log_post)
            post /= post.sum()

            alpha_grid = grid[:, None] * mu[None, :]
            alpha_grid = np.maximum(alpha_grid, 1e-8)
            exc_grid = _exceedance_copula(alpha_grid)

            return float(np.dot(post, exc_grid))

        def _temp_scale(p, temp):
            lp = np.log(np.clip(p, 1e-30, None)) / temp
            lp -= lp.max()
            out = np.exp(lp)
            return out / out.sum()

        def _fit_dirichlet_mle(log_p_sum, N, prior_strength=1.0, max_iter=50, tol=1e-4):
            K = len(log_p_sum)
            if prior_strength > 0:
                pseudo_log = np.full(K, np.log(1.0 / K))
                suff_stats = (log_p_sum + prior_strength * pseudo_log) / (N + prior_strength)
            else:
                suff_stats = log_p_sum / N
            psi_K = math.log(K) - 0.5 / K
            alpha = np.maximum(_inverse_digamma(suff_stats + psi_K), 0.1)
            for _ in range(max_iter):
                a_new = _inverse_digamma(float(_digamma(np.sum(alpha))) + suff_stats)
                a_new = np.maximum(a_new, 1e-8)
                if np.max(np.abs(a_new - alpha)) < tol:
                    return a_new
                alpha = a_new
            return alpha

        def _batch_mle_trajectories(questions, prior_strength=1.0, temp=1.0):
            n_q = len(questions)
            n_max_q = len(questions[0]["probs"])
            K = 4
            eps_smooth = 1e-3
            pseudo_log = np.log(1.0 / K)

            first_argmax = np.zeros(n_q, dtype=int)
            batch_suff, batch_qi = [], []

            for qi in range(n_q):
                plist = questions[qi]["probs"]
                log_p_sum = np.zeros(K)
                for n in range(n_max_q):
                    p = np.asarray(plist[n], dtype=float)
                    if temp > 1.0:
                        p = _temp_scale(p, temp)
                    p_smooth = eps_smooth / K + (1.0 - eps_smooth) * p
                    log_p_sum += np.log(p_smooth)
                    if n == 0:
                        first_argmax[qi] = int(np.argmax(p))
                    else:
                        N = n + 1
                        suff = (log_p_sum + prior_strength * pseudo_log) / (N + prior_strength)
                        batch_suff.append(suff)
                        batch_qi.append(qi)

            if not batch_suff:
                return [[(0.0, 0)] * n_max_q for _ in range(n_q)]

            B = np.array(batch_suff)
            psi_K = math.log(K) - 0.5 / K
            alpha = np.maximum(_inverse_digamma(B + psi_K), 0.1)
            for _ in range(30):
                asum = alpha.sum(axis=1, keepdims=True)
                a_new = _inverse_digamma(_digamma(asum) + B)
                a_new = np.maximum(a_new, 1e-8)
                if np.max(np.abs(a_new - alpha)) < 1e-4:
                    alpha = a_new
                    break
                alpha = a_new

            exc = _exceedance_copula(alpha)
            leads = np.argmax(alpha, axis=1)

            trajs = [[(0.0, int(first_argmax[qi]))] for qi in range(n_q)]
            for idx, qi in enumerate(batch_qi):
                trajs[qi].append((float(exc[idx]), int(leads[idx])))
            return trajs

        def _compute_trajectory(probs, method, temp):
            traj = []

            if method == "product":
                log_post = np.log(np.full(4, 0.25))
                for n in range(len(probs)):
                    p = np.array(probs[n])
                    raw_lp = np.log(np.clip(p, 1e-30, None))
                    if temp != 1.0:
                        sc = raw_lp / temp
                        sc -= sc.max()
                        lk = np.exp(sc)
                        lk /= lk.sum()
                        log_post += np.log(np.clip(lk, 1e-30, None))
                    else:
                        log_post += raw_lp
                    lp = log_post - log_post.max()
                    post = np.exp(lp)
                    post /= post.sum()
                    traj.append((float(post.max()), int(np.argmax(post))))

            elif method == "mle":
                K = 4
                eps_smooth = 1e-3
                log_p_sum = np.zeros(K)
                for n in range(len(probs)):
                    p = np.array(probs[n], dtype=float)
                    if temp > 1.0:
                        p = _temp_scale(p, temp)
                    p_smooth = eps_smooth / K + (1.0 - eps_smooth) * p
                    log_p_sum += np.log(p_smooth)
                    N = n + 1
                    if N < 2:
                        traj.append((0.0, int(np.argmax(p))))
                        continue
                    alpha = _fit_dirichlet_mle(log_p_sum, N)
                    traj.append((_exceedance_copula(alpha), int(np.argmax(alpha))))

            elif method in ("mom", "mom_bayes"):
                K = 4
                p_sum = np.zeros(K)
                p_sq_sum = np.zeros(K)
                for n in range(len(probs)):
                    p = np.array(probs[n], dtype=float)
                    if temp > 1.0:
                        p = _temp_scale(p, temp)
                    p_sum += p
                    p_sq_sum += p * p
                    N = n + 1
                    if N < 2:
                        traj.append((0.0, int(np.argmax(p))))
                        continue
                    mu = p_sum / N
                    var = (p_sq_sum / N - mu * mu) * N / (N - 1)
                    var = np.maximum(var, 0.0)
                    var_sum = var.sum()
                    mu_var_sum = (mu * (1.0 - mu)).sum()
                    if method == "mom":
                        alpha = _mom_estimate_alpha(mu, var_sum, mu_var_sum, N, K)
                        traj.append((_exceedance_copula(alpha), int(np.argmax(mu))))
                    else:
                        exc = _bayesian_mom_exceedance(mu, var_sum, mu_var_sum, N, K)
                        traj.append((exc, int(np.argmax(mu))))

            else:  # sum
                alpha = np.ones(4)
                for n in range(len(probs)):
                    p = np.array(probs[n])
                    if temp > 1.0:
                        p = _temp_scale(p, temp)
                    alpha = alpha + p
                    traj.append((_exceedance_copula(alpha), int(np.argmax(alpha))))

            return traj

        def _augmented_answer(q, alpha_base, esc_key):
            ep = q.get(f"{esc_key}_probs")
            if ep is not None:
                return int(np.argmax(alpha_base + np.array(ep)))
            return q.get(esc_key)

        def _sweep_tau(questions, trajs, tau, n_q, esc1_run, esc2_run):
            ns, corrects, caps = [], [], []
            esc1_c, esc2_c = [], []
            for i, traj in enumerate(trajs):
                q = questions[i]
                if tau == 0.0:
                    answer, n_stop, capped_q = traj[0][1], 1, False
                else:
                    n_stop, capped_q = len(traj), True
                    answer = traj[-1][1]
                    for n, (conf, argmax_a) in enumerate(traj):
                        if conf > tau:
                            answer, n_stop, capped_q = argmax_a, n + 1, False
                            break
                ok = (answer == q["correct"])
                ns.append(n_stop)
                corrects.append(ok)
                caps.append(capped_q)
                if capped_q:
                    alpha_base = np.ones(4)
                    for p in q["probs"]:
                        alpha_base += np.array(p)
                    a1 = _augmented_answer(q, alpha_base, "esc1")
                    a2 = _augmented_answer(q, alpha_base, "esc2")
                    esc1_c.append((a1 == q["correct"]) if a1 is not None else ok)
                    esc2_c.append((a2 == q["correct"]) if a2 is not None else ok)
                else:
                    esc1_c.append(ok)
                    esc2_c.append(ok)
            return {
                "avg_n": float(np.mean(ns)),
                "acc": sum(corrects) / n_q,
                "cap": sum(caps) / n_q,
                "esc1": sum(esc1_c) / n_q if esc1_run else None,
                "esc2": sum(esc2_c) / n_q if esc2_run else None,
            }

        st.info("No precomputed cache found. Computing live (this may take ~10 min)... "
                "Run `python dashboard/precompute_adaptive.py` to create a cache.")

        # --- Compute raw trajectories (T=1, no calibration) ---
        for mn, mdata in model_data.items():
            qs = mdata["questions"]
            mdata["prod_trajs"] = [
                _compute_trajectory(q["probs"], "product", 1.0) for q in qs
            ]
            mdata["sum_trajs"] = [
                _compute_trajectory(q["probs"], "sum", 1.0) for q in qs
            ]
            mdata["mle_trajs"] = _batch_mle_trajectories(qs, temp=1.0)
            mdata["mom_trajs"] = [
                _compute_trajectory(q["probs"], "mom", 1.0) for q in qs
            ]
            mdata["mom_bayes_trajs"] = [
                _compute_trajectory(q["probs"], "mom_bayes", 1.0) for q in qs
            ]

        # --- Raw tau sweep ---
        for mn, mdata in model_data.items():
            qs, n_q_m = mdata["questions"], mdata["n_q"]
            er1, er2 = mdata["esc1_run"], mdata["esc2_run"]
            for _, mkey in METHODS:
                results = []
                for tau in tau_values:
                    r = _sweep_tau(qs, mdata[f"{mkey}_trajs"], tau, n_q_m, er1, er2)
                    r["tau"] = tau
                    results.append(r)
                mdata[f"{mkey}_results"] = results

        model_names = list(model_data.keys())

    # ===================================================================
    # SECTION 1: RAW RESULTS (T = 1)
    # ===================================================================
    st.subheader("Raw Results (T = 1)")
    st.caption(
        "No temperature calibration. All methods operate on the model's raw "
        "logprob vectors. This is the honest baseline — no knobs turned."
    )

    # --- Cross-method highlighting: best acc (teal), worst acc if spread >1.5pp (rose) ---
    def _make_cross_method_highlight(model_data_src, model_names_src, methods_src,
                                     tau_vals):
        """For each (model, tau): highlight best and worst accuracy across methods."""
        acc_range = {}
        for mn in model_names_src:
            for ti, tau in enumerate(tau_vals):
                if tau == 0.0:
                    continue
                vals = []
                for _, mkey in methods_src:
                    res = model_data_src[mn].get(f"{mkey}_results", [])
                    if ti < len(res):
                        v = res[ti].get("acc")
                        if v is not None:
                            vals.append(v)
                if len(vals) >= 2:
                    lo, hi = min(vals), max(vals)
                    acc_range[(mn, ti)] = (lo, hi)

        def _hl(mn, ti, metric, value, skip=False):
            if skip or value is None or metric != "acc":
                return ""
            key = (mn, ti)
            if key not in acc_range:
                return ""
            lo, hi = acc_range[key]
            spread = hi - lo
            if spread < 0.005:
                return ""
            if abs(value - hi) < 1e-6:
                return ' style="background:rgba(42,140,143,0.18);font-weight:700"'
            if abs(value - lo) < 1e-6 and spread >= 0.015:
                return ' style="background:rgba(202,74,122,0.12)"'
            return ""
        return _hl

    _cs = _make_cross_method_highlight(
        model_data, model_names, METHODS, tau_values,
    )

    _at_css = """<style>
    .at { border-collapse: collapse; width: 100%%; font-family: Inter, sans-serif; font-size: 13px; }
    .at th { padding: 6px 10px; text-align: center; border-bottom: 2px solid #2A8C8F;
        color: #2C3E50; font-weight: 600; }
    .at td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #E5E0DB; }
    .at tr:hover { background: #F5F3F1; }
    .at .bl td { border-top: 2px solid #ccc; font-style: italic; color: #8B95A1; }
    .at .mb { border-left: 3px solid #2A8C8F; }
    </style>"""

    # --- Reusable table renderer (used by both raw and calibrated sections) ---
    def _render_tau_tables(data_src, names_src, methods_src, tau_vals,
                           hl_fn, css, n_max_val, desc_override=None,
                           show_temp=False):
        """Render τ-sweep tables: one per method, models side by side."""
        for method_label, mkey in methods_src:
            if desc_override and mkey in desc_override:
                desc = desc_override[mkey]
            else:
                desc = {"prod": "multiply likelihoods, raw",
                        "sum": "Dirichlet pseudo-counts",
                        "mle": "fit α via Minka MLE",
                        "mom": "variance-ratio α₀ estimate",
                        "mom_bayes": "Bayesian wrapper on MoM"}[mkey]
            st.markdown(f"**{method_label}** ({desc})")
            html = css + '<table class="at"><tr><th rowspan="2" class="mb">\u03c4</th>'
            for mn in names_src:
                ncols = 3
                if data_src[mn]["has_esc1"]:
                    ncols += 1
                if data_src[mn]["has_esc2"]:
                    ncols += 1
                html += f'<th class="mb" colspan="{ncols}">{mn}</th>'
            html += "</tr><tr>"
            for mi, mn in enumerate(names_src):
                first = ' class="mb"' if True else ''
                html += f"<th{first}>N\u0304</th><th>acc</th><th>cap</th>"
                if data_src[mn]["has_esc1"]:
                    html += "<th>+CoT</th>"
                if data_src[mn]["has_esc2"]:
                    html += "<th>+think</th>"
            html += "</tr>"
            for ti, tau in enumerate(tau_vals):
                is_zero = (tau == 0.0)
                html += f'<tr><td class="mb">{tau:.2f}</td>'
                for mn in names_src:
                    r = data_src[mn][f"{mkey}_results"][ti]
                    html += f'<td class="mb">{r["avg_n"]:.2f}</td>'
                    html += f'<td{hl_fn(mn, ti, "acc", r["acc"], is_zero)}>{r["acc"]:.1%}</td>'
                    html += f'<td{hl_fn(mn, ti, "cap", r["cap"], is_zero)}>{r["cap"]:.1%}</td>'
                    if data_src[mn]["has_esc1"]:
                        v = r["esc1"]
                        if v is not None:
                            delta = v - r["acc"]
                            d_str = f"+{delta:.1%}" if delta > 1e-9 else "-"
                            html += f'<td{hl_fn(mn, ti, "esc1", v, is_zero)}>{d_str}</td>'
                        else:
                            html += "<td>-</td>"
                    if data_src[mn]["has_esc2"]:
                        v = r["esc2"]
                        if v is not None:
                            delta = v - r["acc"]
                            d_str = f"+{delta:.1%}" if delta > 1e-9 else "-"
                            html += f'<td{hl_fn(mn, ti, "esc2", v, is_zero)}>{d_str}</td>'
                        else:
                            html += "<td>-</td>"
                html += "</tr>"
            html += '<tr class="bl"><td class="mb">all</td>'
            for mn in names_src:
                html += f'<td class="mb">{n_max_val}</td><td>{data_src[mn]["baseline_acc"]:.1%}</td><td>-</td>'
                if data_src[mn]["has_esc1"]:
                    html += "<td>-</td>"
                if data_src[mn]["has_esc2"]:
                    html += "<td>-</td>"
            html += "</tr></table>"
            st.markdown(html, unsafe_allow_html=True)

    def _render_tau_by_model(data_src, names_src, methods_src, tau_vals,
                            hl_fn, css, n_max_val):
        """Render τ-sweep tables: one per model, methods as column groups."""
        for mn in names_src:
            st.markdown(f"##### {mn}")
            has_esc1 = data_src[mn]["has_esc1"]
            has_esc2 = data_src[mn]["has_esc2"]

            sub_cols = 3
            if has_esc1:
                sub_cols += 1
            if has_esc2:
                sub_cols += 1

            html = css + '<table class="at"><tr><th rowspan="2" class="mb">\u03c4</th>'
            for method_label, mkey in methods_src:
                mcolor = MCOLORS[mkey]
                html += f'<th class="mb" colspan="{sub_cols}" style="border-bottom:2px solid {mcolor};color:{mcolor}">{method_label}</th>'
            html += "</tr><tr>"
            for _, mkey in methods_src:
                html += '<th class="mb">N\u0304</th><th>acc</th><th>cap</th>'
                if has_esc1:
                    html += "<th>+CoT</th>"
                if has_esc2:
                    html += "<th>+thk</th>"
            html += "</tr>"

            for ti, tau in enumerate(tau_vals):
                is_zero = (tau == 0.0)
                html += f'<tr><td class="mb">{tau:.2f}</td>'
                for _, mkey in methods_src:
                    r = data_src[mn].get(f"{mkey}_results", [])
                    if ti < len(r):
                        r = r[ti]
                    else:
                        html += f'<td class="mb" colspan="{sub_cols}">-</td>'
                        continue
                    html += f'<td class="mb">{r["avg_n"]:.2f}</td>'
                    html += f'<td{hl_fn(mn, ti, "acc", r["acc"], is_zero)}>{r["acc"]:.1%}</td>'
                    html += f'<td{hl_fn(mn, ti, "cap", r["cap"], is_zero)}>{r["cap"]:.1%}</td>'
                    if has_esc1:
                        v = r.get("esc1")
                        if v is not None:
                            delta = v - r["acc"]
                            d_str = f"+{delta:.1%}" if delta > 1e-9 else "-"
                            html += f'<td{hl_fn(mn, ti, "esc1", v, is_zero)}>{d_str}</td>'
                        else:
                            html += "<td>-</td>"
                    if has_esc2:
                        v = r.get("esc2")
                        if v is not None:
                            delta = v - r["acc"]
                            d_str = f"+{delta:.1%}" if delta > 1e-9 else "-"
                            html += f'<td{hl_fn(mn, ti, "esc2", v, is_zero)}>{d_str}</td>'
                        else:
                            html += "<td>-</td>"
                html += "</tr>"

            html += f'<tr class="bl"><td class="mb">all</td>'
            for _, mkey in methods_src:
                html += f'<td class="mb">{n_max_val}</td><td>{data_src[mn]["baseline_acc"]:.1%}</td><td>-</td>'
                if has_esc1:
                    html += "<td>-</td>"
                if has_esc2:
                    html += "<td>-</td>"
            html += "</tr></table>"
            st.markdown(html, unsafe_allow_html=True)

    _render_tau_by_model(model_data, model_names, METHODS, tau_values,
                         _cs, _at_css, n_max)
    st.caption(
        "Accuracy highlighting: **teal** = best method at each \u03c4 (spread \u2265 0.5 pp), "
        "**rose** = worst (spread \u2265 1.5 pp). Other metrics unhighlighted."
    )

    # --- Method explanations ---
    with st.expander("How it works — Product (multiply likelihoods)"):
        st.markdown(
            "**Model:** Each shuffle permutation gives a probability vector "
            "[P(A), P(B), P(C), P(D)] from logprobs. Treat these as likelihood "
            "functions and apply Bayes' rule with a uniform prior:\n\n"
            "$$P(\\text{answer}=k \\mid \\text{samples}_{1..N}) \\propto "
            "\\frac{1}{4} \\times \\prod_{n=1}^{N} p_k^{(n)}$$\n\n"
            "In log-space: add log-probs, subtract max for numerical stability, "
            "exponentiate, normalise.\n\n"
            "**Stopping criterion:** max P(k | data) > τ — the posterior probability "
            "of the leading answer exceeds the threshold.\n\n"
            "**Temperature scaling:** Raw per-query logprobs are overconfident "
            "(MSP \u2248 0.91, ECE = 0.186). The posterior concentrates in 1\u20132 samples, "
            "making the framework useless. Temperature T deflates confidence: "
            "new\\_probs = softmax(log(probs) / T). T = 3.0 is ECE-optimal "
            "(0.186 \u2192 0.012). The raw section above shows T=1 (no calibration); "
            "the calibrated section below finds optimal T per method.\n\n"
            "**Tradeoff:** Mathematically correct Bayesian update *if* the logprobs "
            "are well-calibrated. In practice they aren't, so temperature helps \u2014 "
            "but it's an extra hyperparameter to tune.\n\n"
            "**Escalation:** Capped questions don't replace the answer — they "
            "**augment** the Sum-style Dirichlet (α = 1 + Σ pₖ) with the "
            "escalation mode's probability vector: α\\_aug = α + p\\_esc. "
            "The augmented argmax combines direct evidence with CoT/think evidence "
            "rather than discarding the accumulated permutation data."
        )
    with st.expander("How it works — Sum (Dirichlet pseudo-counts)"):
        st.markdown(
            "**Model:** Instead of multiplying likelihood vectors, accumulate them "
            "as fractional pseudo-counts in a Dirichlet distribution:\n\n"
            "$$\\alpha_k = 1 + \\sum_{n=1}^{N} p_k^{(n)}$$\n\n"
            "Each observation distributes one pseudo-count proportionally across "
            "categories. The prior is uniform (α₀ = 1 per category).\n\n"
            "**Stopping criterion:** Exceedance probability > τ — the probability "
            "that the leading category is truly the best, computed via damped "
            "Gaussian-copula approximation:\n\n"
            "$$P(\\theta_{\\text{lead}} > \\text{all others}) \\approx "
            "\\prod_j p_j + d(K) \\sum_{j<k} \\rho_{jk}\\,\\phi(a_j)\\,\\phi(a_k)"
            "\\prod_{l \\neq j,k} p_l$$\n\n"
            "where p_j = 1 − I₀.₅(α_lead, α_j) are exact pairwise Beta exceedances, "
            "a_j = Φ⁻¹(p_j) are probit transforms, ρ_jk are Dirichlet-derived "
            "correlations, and d(K) is a calibrated damping factor. RMSE ~0.005 "
            "for K=4 (vs ~0.04 for Bonferroni).\n\n"
            "**No temperature needed.** Evidence accumulates linearly (not "
            "exponentially), so overconfident logprobs don't collapse the posterior. "
            "Product: 0.9 × 0.9 × 0.9 = 0.73 posterior weight after 3 samples. "
            "Sum: α\\_lead = 1 + 3×0.9 = 3.7, Σα = 7.0 — much less concentrated.\n\n"
            "**Tradeoff:** Less statistically efficient — needs more "
            "samples to reach the same confidence for genuinely easy questions. "
            "But doesn't break on miscalibrated inputs.\n\n"
            "**Escalation:** Capped questions augment the Dirichlet: "
            "α\\_aug = α\\_cap + p\\_esc. The CoT/think probability vector adds "
            "evidence to the existing posterior rather than replacing it. This is "
            "Luigi's design — it can never make things worse since the accumulated "
            "direct evidence is preserved."
        )
    with st.expander("How it works — Dirichlet MLE (fit from data)"):
        st.markdown(
            "**Model:** Assume the N observed probability vectors are drawn from a "
            "Dirichlet distribution Dir(α₁, …, α₄). Fit α via **maximum likelihood** "
            "(Minka 2000 fixed-point iteration):\n\n"
            "$$\\hat{\\alpha} = \\arg\\max_\\alpha \\sum_{n=1}^{N} "
            "\\log \\text{Dir}(p^{(n)} \\mid \\alpha)$$\n\n"
            "The algorithm iterates: compute target values from the mean of "
            "log-probabilities and the current total concentration, then invert "
            "the digamma function to update each α_k. Converges in ~10–20 iterations.\n\n"
            "**Stopping criterion:** Damped Gaussian-copula exceedance > τ (same "
            "as Sum), but computed on the MLE-fitted α instead of pseudo-count α.\n\n"
            "**Regularisation:** A uniform pseudo-observation (prior\\_strength=1.0) "
            "is added to the sufficient statistics, plus label smoothing (ε=10⁻³) "
            "on the raw probability vectors. This stabilises low-N fits and prevents "
            "log(0) without distorting the MLE at higher N.\n\n"
            "**Key difference from Sum:** Sum sets α_k = 1 + Σ p_k — a heuristic "
            "that treats probability vectors as fractional counts. MLE fits the "
            "actual Dirichlet shape from the *variation* across permutations. "
            "If all 5 vectors are nearly identical, MLE estimates high concentration "
            "(very confident). If they vary, MLE estimates low concentration (uncertain). "
            "Sum can't distinguish 'five identical vectors' from 'five diverse vectors "
            "that happen to average the same.'\n\n"
            "**No temperature needed.** The concentration parameter is estimated from "
            "data, not assumed. The MLE naturally handles overconfident logprobs by "
            "fitting the observed distribution shape rather than trusting the raw "
            "probability values.\n\n"
            "**Caveat:** Needs N ≥ 2 observations (can't fit a distribution from one "
            "point), so confidence is always 0 at N=1. With small N, the MLE can be "
            "noisy. Luigi's concern: this is the formally correct approach, but the "
            "pseudo-count approximation may work just as well in practice with our "
            "sample sizes (N ≤ 10)."
        )
    with st.expander("How it works — MoM (Method of Moments)"):
        st.markdown(
            "**Model:** For Dirichlet(α₀·μ), the variance of component k is "
            "μₖ(1−μₖ)/(α₀+1). The pooled variance ratio R̂ = Σₖ s²ₖ / Σₖ μ̂ₖ(1−μ̂ₖ) "
            "estimates 1/(α₀+1), giving:\n\n"
            "$$\\hat{\\alpha}_0 = \\frac{1 - \\hat{R}}{\\hat{R}}, \\qquad "
            "\\hat{\\alpha} = \\hat{\\alpha}_0 \\cdot \\hat{\\mu}$$\n\n"
            "**Stopping criterion:** Copula exceedance on the estimated α (same as "
            "Sum and MLE). Needs N ≥ 2 — can't estimate variance from one sample.\n\n"
            "**Key advantage:** One-line formula. No iterative fitting (unlike MLE), "
            "no temperature (unlike Product). The pooled ratio naturally downweights "
            "components near 0 or 1 where variance estimates are noisy.\n\n"
            "**Key limitation:** Point estimate with no error bars. At N=2, the sample "
            "variance has 1 degree of freedom — the 95% CI on s² spans a factor of "
            "~1000. The stopping criterion doesn't know how uncertain it is about "
            "its own certainty. See MoM+Bayes for the principled fix."
        )
    with st.expander("How it works — MoM + Bayes (Bayesian wrapper)"):
        st.markdown(
            "**Model:** Same variance ratio R̂ as MoM, but instead of trusting the "
            "point estimate, put a Gamma(2, 5) prior on α₀ and compute a posterior:\n\n"
            "$$P(\\alpha_0 \\mid \\hat{R}, N) \\propto "
            "f_{\\chi^2}\\!\\left(\\frac{\\hat{R}}{R(\\alpha_0)} \\cdot \\nu;\\; \\nu\\right) "
            "\\cdot \\pi(\\alpha_0)$$\n\n"
            "where R(α₀) = 1/(α₀+1), ν = K(N−1) degrees of freedom, and π is the "
            "Gamma prior.\n\n"
            "**Stopping criterion:** Marginalized exceedance — integrate the copula "
            "exceedance over the α₀ posterior on an 80-point log-spaced grid:\n\n"
            "$$P(\\text{leader best}) = \\sum_g P(\\text{leader best} \\mid \\alpha_{0,g}, "
            "\\hat{\\mu}) \\cdot P(\\alpha_{0,g} \\mid \\hat{R}, N) \\cdot \\Delta g$$\n\n"
            "**Why it matters for adaptive stopping:** At small N (2–3), the posterior "
            "on α₀ is wide → the exceedance is blurred → the system is conservative "
            "(won't stop early on flimsy evidence). At large N (8+), the posterior "
            "tightens → converges to plain MoM. This is exactly the behavior you want: "
            "uncertain about your uncertainty early, decisive later.\n\n"
            "**Prior choice:** Gamma(shape=2, scale=5) → mean=10, mode=5. Says "
            "'models are moderately consistent before seeing data.' Not sensitive — "
            "the data overwhelms the prior by N=4–5."
        )

    # --- Accuracy vs Compute: one chart per model ---
    st.subheader("Accuracy vs Compute")
    st.caption(
        "Left: base accuracy (no escalation) vs avg queries used. "
        "Right: accuracy after escalating capped questions to think mode. "
        "Diamond = all-N baseline (use all permutations, no stopping)."
    )

    for mn in model_names:
        mdata = model_data[mn]
        col_base, col_esc = st.columns(2)

        fig_base = go.Figure()
        fig_esc = go.Figure()
        for method_label, mkey in METHODS:
            color = MCOLORS[mkey]
            results = mdata[f"{mkey}_results"]
            xs = [r["avg_n"] for r in results]
            fig_base.add_trace(go.Scatter(
                x=xs, y=[r["acc"] for r in results],
                mode="lines+markers", name=method_label,
                line=dict(color=color, width=2.5), marker=dict(size=8),
                hovertemplate=(
                    f"{method_label} \u03c4=%{{customdata:.2f}}<br>"
                    "avg_N=%{x:.2f}<br>acc=%{y:.1%}<extra></extra>"
                ),
                customdata=[r["tau"] for r in results],
            ))
            if mdata["has_esc2"]:
                ys = [r["esc2"] for r in results if r["esc2"] is not None]
                if ys:
                    fig_esc.add_trace(go.Scatter(
                        x=xs[:len(ys)], y=ys,
                        mode="lines+markers", name=f"{method_label} +think",
                        line=dict(color=color, width=2.5),
                        marker=dict(size=8, symbol="diamond"),
                        hovertemplate=(
                            f"{method_label} +think \u03c4=%{{customdata:.2f}}<br>"
                            "avg_N=%{x:.2f}<br>acc=%{y:.1%}<extra></extra>"
                        ),
                        customdata=[r["tau"] for r in results[:len(ys)]],
                    ))
        baseline = mdata["baseline_acc"]
        _bl_kwargs = dict(
            x=[n_max], y=[baseline],
            mode="markers", name=f"All-{n_max} baseline",
            marker=dict(size=12, color=GRAY_LIGHT, symbol="diamond"),
            showlegend=True,
        )
        _hl_kwargs = dict(
            y=baseline, line_dash="dash", line_color=GRAY_LIGHT,
            annotation_text=f"all-{n_max} baseline ({baseline:.1%})",
            annotation_position="bottom right",
        )
        _layout_kwargs = dict(
            xaxis_title="avg_N", yaxis_title="Accuracy", height=350,
            yaxis=dict(tickformat=".0%", gridcolor=GRID),
            xaxis=dict(range=[0.5, n_max + 0.5], gridcolor=GRID),
        )
        fig_base.add_trace(go.Scatter(**_bl_kwargs))
        fig_base.add_hline(**_hl_kwargs)
        fig_base.update_layout(**_base_layout(
            title=f"{mn} — Base Accuracy", **_layout_kwargs))

        fig_esc.add_trace(go.Scatter(**_bl_kwargs))
        fig_esc.add_hline(**_hl_kwargs)
        fig_esc.update_layout(**_base_layout(
            title=f"{mn} — +Think Escalation", **_layout_kwargs))

        with col_base:
            st.plotly_chart(_round_hover(fig_base), width="stretch")
        with col_esc:
            st.plotly_chart(_round_hover(fig_esc), width="stretch")

    # ===================================================================
    # SECTION 2: TEMPERATURE CALIBRATION
    # ===================================================================
    st.divider()
    st.subheader("Temperature-Calibrated Results")
    st.caption(
        "Sweep temperature T for **all** methods — not just Product. "
        "Optimal T maximizes acc+escalation on each method's Pareto frontier. "
        "Hypothesis: MoM/Bayes needs less T than Product because concentration "
        "estimation already accounts for overconfidence."
    )

    _ESC_MULTIPLIERS = {
        "CoT": {"Gemma 4": 2.2, "Qwen 3": 2.6, "Qwen 3.5": 3.6},
        "Think": {"Gemma 4": 5.9, "Qwen 3": 7.3, "Qwen 3.5": 20.1},
    }
    _ESC_DIRECT_S = {"Gemma 4": 0.96, "Qwen 3": 1.27, "Qwen 3.5": 1.05}

    esc_mode = st.selectbox(
        "Escalation mode", ["CoT", "Think"], index=1,
        help="Cost of escalation relative to one direct query, measured per model on A100.",
        key="esc_cost_select",
    )
    esc_costs_by_model = _ESC_MULTIPLIERS[esc_mode]

    with st.expander("Cost multipliers (measured on A100)"):
        _cm_h = ('<table style="font-size:12px;border-collapse:collapse">'
                 '<tr><th style="padding:4px 10px">Model</th>'
                 '<th style="padding:4px 10px">Direct</th>'
                 '<th style="padding:4px 10px">CoT \u00d7</th>'
                 '<th style="padding:4px 10px">Think \u00d7</th></tr>')
        for _mn in model_names:
            _d = _ESC_DIRECT_S.get(_mn, "")
            _c = _ESC_MULTIPLIERS["CoT"].get(_mn, "")
            _t = _ESC_MULTIPLIERS["Think"].get(_mn, "")
            _cm_h += (f'<tr><td style="padding:4px 10px">{_mn}</td>'
                      f'<td style="padding:4px 10px;text-align:center">{_d}s</td>'
                      f'<td style="padding:4px 10px;text-align:center">{_c}\u00d7</td>'
                      f'<td style="padding:4px 10px;text-align:center">{_t}\u00d7</td></tr>')
        _cm_h += '</table>'
        st.markdown(_cm_h, unsafe_allow_html=True)
        st.caption(
            "Multipliers = mean inference\\_time\\_s / direct mean, from 50-question "
            "timing runs. `total\\_cost = avg\\_N + cap\\_rate \u00d7 esc\\_multiplier`."
        )

    def _pareto_from_grid(grid, esc_cost_val):
        """Pareto frontier, computing total_cost on the fly."""
        enriched = [
            {**r, "total_cost": r["avg_n"] + r["cap_rate"] * esc_cost_val}
            for r in grid
        ]
        pareto = []
        by_cost = sorted(enriched, key=lambda r: (r["total_cost"], -r["acc_esc"]))
        best = -1.0
        for r in by_cost:
            if r["acc_esc"] > best:
                best = r["acc_esc"]
                pareto.append(r)
        return pareto, enriched

    cal_opt: dict = {}
    cal_pareto: dict = {}
    cal_model_data: dict = {}

    # Use cached grid sweeps (from precompute script or session state)
    _have_cal_grids = bool(cal_grids_cached)
    if not _have_cal_grids:
        _cal_cache_key = (
            tuple(sorted((mn, model_data[mn]["n_q"]) for mn in model_names)),
            n_max,
        )
        if (st.session_state.get("cal_cache_key") == _cal_cache_key
                and "cal_grids" in st.session_state):
            cal_grids_cached = st.session_state.cal_grids
            _have_cal_grids = True

    if _have_cal_grids:
        cal_grids = cal_grids_cached
        for (mkey, mn), grid in cal_grids.items():
            _mn_esc = esc_costs_by_model.get(mn, 7.1)
            pareto, enriched = _pareto_from_grid(grid, _mn_esc)
            cal_pareto[(mkey, mn)] = pareto
            cal_grids[(mkey, mn)] = enriched
            if pareto:
                cal_opt[(mkey, mn)] = max(pareto, key=lambda r: r["acc_esc"])

        # --- Optimal T table ---
        st.markdown("**Optimal Temperature per Method \u00d7 Model**")
        html = _at_css + '<table class="at"><tr><th class="mb">Method</th>'
        for mn in model_names:
            html += f'<th class="mb" colspan="4">{mn}</th>'
        html += '</tr><tr><th class="mb"></th>'
        for mn in model_names:
            html += '<th class="mb">T*</th><th>\u03c4*</th><th>acc_esc</th><th>avg_N</th>'
        html += '</tr>'
        for method_label, mkey in METHODS:
            html += f'<tr><td class="mb"><b>{method_label}</b></td>'
            for mn in model_names:
                b = cal_opt.get((mkey, mn))
                if b:
                    html += (f'<td class="mb"><b>{b["T"]:.1f}</b></td>'
                             f'<td>{b["tau"]:.2f}</td>'
                             f'<td>{b["acc_esc"]:.1%}</td>'
                             f'<td>{b["avg_n"]:.1f}</td>')
                else:
                    html += '<td class="mb" colspan="4">-</td>'
            html += '</tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
        st.caption(
            "T* = temperature that maximises acc+escalation on the Pareto frontier. "
            "\u03c4* = corresponding stopping threshold. "
            "Higher T softens per-query logprobs before posterior aggregation."
        )

        # --- Extract calibrated τ-sweep from grid at optimal T ---
        for mn in model_names:
            mdata = model_data[mn]
            cal_md = {
                "questions": mdata["questions"], "n_q": mdata["n_q"],
                "baseline_acc": mdata["baseline_acc"],
                "has_esc1": mdata["has_esc1"], "has_esc2": mdata["has_esc2"],
                "esc1_run": mdata["esc1_run"], "esc2_run": mdata["esc2_run"],
            }
            for _, mkey in METHODS:
                opt_T = cal_opt.get((mkey, mn), {}).get("T", 1.0)
                grid = cal_grids_cached.get((mkey, mn), [])
                results = []
                for tau in tau_values_cal:
                    match = next(
                        (r for r in grid
                         if abs(r["T"] - opt_T) < 0.01 and abs(r["tau"] - tau) < 0.001),
                        None,
                    )
                    if match:
                        results.append({
                            "tau": tau, "avg_n": match["avg_n"],
                            "acc": match["acc"], "cap": match["cap_rate"],
                            "esc1": match.get("esc1"),
                            "esc2": match.get("esc2"),
                        })
                    else:
                        results.append({
                            "tau": tau, "avg_n": 0, "acc": 0, "cap": 0,
                            "esc1": None, "esc2": None,
                        })
                cal_md[f"{mkey}_results"] = results
            cal_model_data[mn] = cal_md

        # --- Calibrated \u03c4-sweep tables ---
        cal_hl = _make_cross_method_highlight(
            cal_model_data, model_names, METHODS, tau_values_cal,
        )
        cal_descs = {}
        for _, mkey in METHODS:
            ts = [f"{cal_opt.get((mkey, mn), {}).get('T', 1.0):.1f}"
                  for mn in model_names]
            cal_descs[mkey] = f"T* = {', '.join(ts)}"

        _render_tau_by_model(cal_model_data, model_names, METHODS, tau_values_cal,
                             cal_hl, _at_css, n_max)
        st.caption(
            "Accuracy highlighting: **teal** = best method at each \u03c4 (spread \u2265 0.5 pp), "
            "**rose** = worst (spread \u2265 1.5 pp). Other metrics unhighlighted."
        )

        # --- Calibrated Accuracy vs Compute ---
        st.markdown("**Accuracy vs Compute (at optimal T)**")
        st.caption(
            "Same as above but using each method's optimal temperature. "
            "Compare how methods trade off compute for accuracy after calibration."
        )
        for mn in model_names:
            cmd = cal_model_data[mn]
            col_base_c, col_esc_c = st.columns(2)

            fig_base_c = go.Figure()
            fig_esc_c = go.Figure()
            for method_label, mkey in METHODS:
                color = MCOLORS[mkey]
                results = cmd[f"{mkey}_results"]
                opt_T = cal_opt.get((mkey, mn), {}).get("T", 1.0)
                xs = [r["avg_n"] for r in results]
                fig_base_c.add_trace(go.Scatter(
                    x=xs, y=[r["acc"] for r in results],
                    mode="lines+markers",
                    name=f"{method_label} T={opt_T:.1f}",
                    line=dict(color=color, width=2.5), marker=dict(size=8),
                    hovertemplate=(
                        f"{method_label} T={opt_T:.1f} "
                        "\u03c4=%{customdata:.2f}<br>"
                        "avg_N=%{x:.2f}<br>acc=%{y:.1%}<extra></extra>"
                    ),
                    customdata=[r["tau"] for r in results],
                ))
                if cmd["has_esc2"]:
                    ys = [r["esc2"] for r in results if r["esc2"] is not None]
                    if ys:
                        fig_esc_c.add_trace(go.Scatter(
                            x=xs[:len(ys)], y=ys,
                            mode="lines+markers",
                            name=f"{method_label} +think T={opt_T:.1f}",
                            line=dict(color=color, width=2.5),
                            marker=dict(size=8, symbol="diamond"),
                            hovertemplate=(
                                f"{method_label} +think T={opt_T:.1f} "
                                "\u03c4=%{customdata:.2f}<br>"
                                "avg_N=%{x:.2f}<br>acc=%{y:.1%}<extra></extra>"
                            ),
                            customdata=[r["tau"] for r in results[:len(ys)]],
                        ))
            baseline = cmd["baseline_acc"]
            _bl_c = dict(
                x=[n_max], y=[baseline],
                mode="markers", name=f"All-{n_max} baseline",
                marker=dict(size=12, color=GRAY_LIGHT, symbol="diamond"),
            )
            _hl_c = dict(
                y=baseline, line_dash="dash", line_color=GRAY_LIGHT,
                annotation_text=f"all-{n_max} baseline ({baseline:.1%})",
                annotation_position="bottom right",
            )
            _lay_c = dict(
                xaxis_title="avg_N", yaxis_title="Accuracy", height=350,
                yaxis=dict(tickformat=".0%", gridcolor=GRID),
                xaxis=dict(range=[0.5, n_max + 0.5], gridcolor=GRID),
            )
            fig_base_c.add_trace(go.Scatter(**_bl_c))
            fig_base_c.add_hline(**_hl_c)
            fig_base_c.update_layout(**_base_layout(
                title=f"{mn} \u2014 Base (calibrated)", **_lay_c))

            fig_esc_c.add_trace(go.Scatter(**_bl_c))
            fig_esc_c.add_hline(**_hl_c)
            fig_esc_c.update_layout(**_base_layout(
                title=f"{mn} \u2014 +Think (calibrated)", **_lay_c))

            with col_base_c:
                st.plotly_chart(_round_hover(fig_base_c), width="stretch")
            with col_esc_c:
                st.plotly_chart(_round_hover(fig_esc_c), width="stretch")

        # --- Pareto frontier charts (all methods sweep T) ---
        st.markdown("**Pareto Frontiers (all methods sweep T \u00d7 \u03c4)**")
        st.caption(
            "Each point is a (T, \u03c4) configuration. Stars = Pareto-optimal points "
            "(no other config achieves higher accuracy at equal or lower cost). "
            "Faint dots = dominated configurations. Cost = avg\\_N + cap\\_rate \u00d7 esc\\_multiplier."
        )

        def _add_method_traces(fig, grid, pareto, color, name):
            p_keys = {(r["T"], r["tau"]) for r in pareto}
            non_p = [r for r in grid if (r["T"], r["tau"]) not in p_keys]
            t_fmt = f"{name} T=%{{customdata[0]:.1f}}, "
            if non_p:
                fig.add_trace(go.Scatter(
                    x=[r["total_cost"] for r in non_p],
                    y=[r["acc_esc"] for r in non_p],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.10),
                    hovertemplate=(
                        t_fmt + "\u03c4=%{customdata[1]:.2f}<br>"
                        "cost=%{x:.1f} \u00b7 acc+esc=%{y:.1%}<br>"
                        "avg_N=%{customdata[2]:.1f} \u00b7 cap=%{customdata[3]:.0%}"
                        "<extra></extra>"
                    ),
                    customdata=[[r["T"], r["tau"], r["avg_n"], r["cap_rate"]]
                                for r in non_p],
                    name=f"{name} (all)", showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x=[r["total_cost"] for r in pareto],
                y=[r["acc_esc"] for r in pareto],
                mode="lines+markers",
                marker=dict(size=8, color=color, symbol="star"),
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    t_fmt + "\u03c4=%{customdata[1]:.2f}<br>"
                    "cost=%{x:.1f} \u00b7 acc+esc=%{y:.1%}<br>"
                    "avg_N=%{customdata[2]:.1f} \u00b7 cap=%{customdata[3]:.0%}"
                    "<extra></extra>"
                ),
                customdata=[[r["T"], r["tau"], r["avg_n"], r["cap_rate"]]
                            for r in pareto],
                name=f"{name} Pareto",
            ))

        for mn in model_names:
            fig2 = go.Figure()
            for method_label, mkey in METHODS:
                grid = cal_grids[(mkey, mn)]
                pareto = cal_pareto[(mkey, mn)]
                _add_method_traces(fig2, grid, pareto, MCOLORS[mkey], method_label)
            baseline = model_data[mn]["baseline_acc"]
            fig2.add_trace(go.Scatter(
                x=[float(n_max)], y=[baseline],
                mode="markers", name=f"All-{n_max} baseline",
                marker=dict(size=12, color=GRAY_LIGHT, symbol="diamond"),
            ))
            fig2.update_layout(**_base_layout(
                title=f"{mn} \u2014 Accuracy vs Total Cost (all methods sweep T)",
                xaxis_title="Total cost per question (base-query units)",
                yaxis_title="Accuracy (with best escalation)",
                height=400,
                yaxis=dict(tickformat=".1%", gridcolor=GRID),
                xaxis=dict(gridcolor=GRID),
            ))
            st.plotly_chart(_round_hover(fig2), width="stretch")

        # --- Cross-method comparison at fixed cost budgets ---
        st.markdown("**Method Comparison at Fixed Cost Budgets**")
        st.caption(
            "Best achievable accuracy for each method at a given compute budget. "
            "Read across to compare methods at the same cost."
        )

        cost_budgets = [2, 3, 4, 5, 7]
        _cmp_css = """<style>
        .cmp { border-collapse: collapse; width: 100%; font-family: Inter, sans-serif; font-size: 12px; }
        .cmp th { padding: 5px 8px; text-align: center; font-weight: 600; font-size: 12px; }
        .cmp td { padding: 5px 8px; text-align: center; border-bottom: 1px solid #E5E0DB; }
        .cmp tr:hover { background: #F5F3F1; }
        .cmp .bl td { border-top: 2px solid #ccc; font-style: italic; color: #8B95A1; }
        .cmp .best { background: rgba(42,140,143,0.18); font-weight: 700; }
        .cmp .worst { background: rgba(202,74,122,0.10); }
        .cmp .delta { font-size: 10px; color: #8B95A1; }
        .cmp .mhdr { font-size: 11px; }
        </style>"""

        for mn in model_names:
            baseline = model_data[mn]["baseline_acc"]
            method_at_budget: dict[str, dict[float, dict]] = {}
            for _, mkey in METHODS:
                pts = cal_pareto.get((mkey, mn), [])
                method_at_budget[mkey] = {}
                for budget in cost_budgets:
                    eligible = [r for r in pts if r["total_cost"] <= budget + 0.05]
                    if eligible:
                        best = max(eligible, key=lambda r: r["acc_esc"])
                        method_at_budget[mkey][budget] = best

            h = _cmp_css + f'<table class="cmp"><tr><th></th>'
            for method_label, mkey in METHODS:
                mcolor = MCOLORS[mkey]
                h += f'<th class="mhdr" style="border-bottom:2px solid {mcolor};color:{mcolor}">{method_label}</th>'
            h += '</tr>'

            for budget in cost_budgets:
                accs = {}
                for _, mkey in METHODS:
                    pt = method_at_budget[mkey].get(budget)
                    if pt:
                        accs[mkey] = pt["acc_esc"]
                if not accs:
                    continue
                best_val = max(accs.values())
                worst_val = min(accs.values()) if len(accs) >= 2 else None
                spread = best_val - worst_val if worst_val is not None else 0

                h += f'<tr><td style="font-weight:600">\u2264{budget}</td>'
                for _, mkey in METHODS:
                    pt = method_at_budget[mkey].get(budget)
                    if pt is None:
                        h += '<td style="color:#ccc">\u2014</td>'
                        continue
                    acc_esc = pt["acc_esc"]
                    gain = acc_esc - baseline
                    cls = ""
                    if spread >= 0.005:
                        if abs(acc_esc - best_val) < 1e-6:
                            cls = ' class="best"'
                        elif abs(acc_esc - worst_val) < 1e-6:
                            cls = ' class="worst"'
                    gain_str = f' <span class="delta">({gain:+.1f}pp)</span>' if abs(gain) >= 0.05 else ""
                    h += f'<td{cls}>{acc_esc:.1%}{gain_str}</td>'
                h += '</tr>'

            h += f'<tr class="bl"><td>all-{n_max}</td>'
            for _ in METHODS:
                h += f'<td>{baseline:.1%}</td>'
            h += '</tr></table>'

            st.markdown(f"**{mn}**")
            st.markdown(h, unsafe_allow_html=True)
        st.caption(
            "Per budget row: **teal** = best method (spread \u2265 0.5 pp), "
            "**rose** = worst. Grey delta = gain over all-N baseline."
        )

        # --- Pareto tables: one section per model, 5 methods side by side ---
        st.markdown("**Pareto Frontier Tables (by model)**")

        _pareto_css = """<style>
        .pt { border-collapse: collapse; width: 100%; font-family: Inter, sans-serif; font-size: 11px; }
        .pt th { padding: 3px 5px; text-align: center; font-weight: 600; font-size: 11px; }
        .pt td { padding: 3px 5px; text-align: center; border-bottom: 1px solid #E5E0DB; font-size: 11px; }
        .pt tr:hover { background: #F5F3F1; }
        .pt .best-row td { font-weight: 700; }
        .pt .mhdr { border-bottom: 2px solid; font-size: 12px; letter-spacing: 0.02em; }
        .pt .winner { font-size: 10px; padding: 1px 5px; border-radius: 3px; margin-left: 4px; }
        </style>"""

        for mn in model_names:
            st.markdown(f"##### {mn}")

            peak_by_method = {}
            efficiency_by_method = {}
            for _, mkey in METHODS:
                pts = cal_pareto.get((mkey, mn), [])
                if pts:
                    best_pt = max(pts, key=lambda r: r["acc_esc"])
                    peak_by_method[mkey] = best_pt["acc_esc"]
                    efficiency_by_method[mkey] = max(
                        r["acc_esc"] / max(r["total_cost"], 0.1) for r in pts
                    )

            best_peak_method = max(peak_by_method, key=peak_by_method.get) if peak_by_method else None
            best_eff_method = max(efficiency_by_method, key=efficiency_by_method.get) if efficiency_by_method else None

            cols = st.columns(len(METHODS))
            for ci, (method_label, mkey) in enumerate(METHODS):
                with cols[ci]:
                    mcolor = MCOLORS[mkey]
                    badges = ""
                    if mkey == best_peak_method:
                        badges += f'<span class="winner" style="background:{mcolor};color:white">peak</span>'
                    if mkey == best_eff_method and mkey != best_peak_method:
                        badges += f'<span class="winner" style="background:{mcolor};color:white">eff</span>'

                    pts = cal_pareto.get((mkey, mn), [])
                    best_acc = max((r["acc_esc"] for r in pts), default=-1)

                    h = _pareto_css + f'<table class="pt">'
                    h += f'<tr><th colspan="7" class="mhdr" style="border-color:{mcolor};color:{mcolor}">'
                    h += f'{method_label}{badges}</th></tr>'
                    h += '<tr><th>T</th><th>\u03c4</th><th>N\u0304</th><th>cap</th>'
                    h += '<th>acc</th><th>+esc</th><th>cost</th></tr>'
                    for r in pts:
                        is_best = abs(r["acc_esc"] - best_acc) < 1e-6
                        cls = ' class="best-row"' if is_best else ""
                        acc_bg = ""
                        if is_best and mkey == best_peak_method:
                            acc_bg = f' style="background:rgba(42,140,143,0.18)"'
                        esc_delta = r["acc_esc"] - r["acc"]
                        esc_str = f"+{esc_delta:.1%}" if esc_delta >= 0.0005 else "-"
                        h += f'<tr{cls}>'
                        h += f'<td>{r["T"]:.1f}</td><td>{r["tau"]:.2f}</td>'
                        h += f'<td>{r["avg_n"]:.1f}</td>'
                        h += f'<td>{r["cap_rate"]:.0%}</td>'
                        h += f'<td{acc_bg}>{r["acc"]:.1%}</td>'
                        h += f'<td>{esc_str}</td>'
                        h += f'<td>{r["total_cost"]:.1f}</td></tr>'
                    if not pts:
                        h += '<tr><td colspan="7">-</td></tr>'
                    h += '</table>'
                    st.markdown(h, unsafe_allow_html=True)
            st.caption("Escalation deltas below 0.05 pp are hidden for clarity.")
    else:
        st.info("No temperature calibration data. "
                "Run `python dashboard/precompute_adaptive.py` or click "
                "**Recalculate Adaptive Cache** in the sidebar.")

    # --- Export results ---
    st.subheader("Export Results")
    import json as _json

    def _make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return float(obj) if hasattr(obj, '__float__') else str(obj)

    export_data: dict = {
        "n_max": n_max,
        "escalation_mode": esc_mode,
        "cost_multipliers": {
            mn: {"CoT": _ESC_MULTIPLIERS["CoT"].get(mn),
                 "Think": _ESC_MULTIPLIERS["Think"].get(mn),
                 "direct_s": _ESC_DIRECT_S.get(mn)}
            for mn in model_names
        },
        "models": {},
    }
    for mn in model_names:
        mdata = model_data[mn]
        m_export: dict = {
            "n_questions": mdata["n_q"],
            "baseline_acc": mdata["baseline_acc"],
            "raw_tau_sweep": {},
        }
        for _, mkey in METHODS:
            m_export["raw_tau_sweep"][mkey] = mdata[f"{mkey}_results"]
        if cal_opt:
            m_export["calibrated"] = {}
            for _, mkey in METHODS:
                opt = cal_opt.get((mkey, mn), {})
                pareto_clean = []
                for r in cal_pareto.get((mkey, mn), []):
                    pareto_clean.append({
                        "T": r.get("T"), "tau": r.get("tau"),
                        "avg_n": r.get("avg_n"), "cap_rate": r.get("cap_rate"),
                        "acc": r.get("acc"), "acc_esc": r.get("acc_esc"),
                        "total_cost": r.get("total_cost"),
                    })
                m_export["calibrated"][mkey] = {
                    "opt_T": opt.get("T"),
                    "opt_tau": opt.get("tau"),
                    "acc_esc": opt.get("acc_esc"),
                    "avg_n": opt.get("avg_n"),
                    "cap_rate": opt.get("cap_rate"),
                    "pareto": pareto_clean,
                }
                if mn in cal_model_data:
                    m_export["calibrated"][mkey]["tau_sweep"] = (
                        cal_model_data[mn][f"{mkey}_results"]
                    )
        export_data["models"][mn] = m_export

    if cal_pareto:
        _budget_levels = [2, 3, 4, 5, 7]
        for mn in model_names:
            budget_data: dict = {}
            for budget in _budget_levels:
                budget_row: dict = {}
                for _, mkey in METHODS:
                    pts = cal_pareto.get((mkey, mn), [])
                    eligible = [r for r in pts if r["total_cost"] <= budget + 0.05]
                    if eligible:
                        best = max(eligible, key=lambda r: r["acc_esc"])
                        budget_row[mkey] = {
                            "acc_esc": best["acc_esc"], "acc": best["acc"],
                            "T": best["T"], "tau": best["tau"],
                            "avg_n": best["avg_n"], "cap_rate": best["cap_rate"],
                            "total_cost": best["total_cost"],
                        }
                budget_data[str(budget)] = budget_row
            export_data["models"][mn]["cost_budget_comparison"] = budget_data

    export_json = _json.dumps(export_data, indent=2, default=_make_serializable)
    export_path = Path(
        r"C:\Users\evama\Dropbox\Family Room\Projects\bayesian-uq"
        r"\paper\results\adaptive_sampling_export.json"
    )

    col_dl, col_save = st.columns(2)
    with col_dl:
        st.download_button(
            "Download results JSON",
            data=export_json,
            file_name="adaptive_sampling_export.json",
            mime="application/json",
        )
    with col_save:
        if st.button("Save to paper/results/", key="save_export"):
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(export_json, encoding="utf-8")
            st.success(f"Saved to `{export_path}`")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Pre-Action Uncertainty Quantification")
st.caption("QuALITY dataset | Multi-model uncertainty quantification")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Progress", "Uncertainty Distributions",
    "Condition Comparison", "Effect Analysis", "Question Explorer",
    "Signal Battery", "Adaptive Sampling",
])

with tab1:
    if HAS_LOCAL_FILES:
        tab_progress()
    else:
        st.info("Progress monitoring requires local result files. "
                "Analysis tabs below use the cached data.")
with tab2:
    tab_distributions()
with tab3:
    tab_comparison()
with tab4:
    tab_effects()
with tab5:
    tab_explorer()
with tab6:
    tab_signals()
with tab7:
    tab_adaptive()
