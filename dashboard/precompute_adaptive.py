"""Precompute adaptive sampling results for the Streamlit dashboard.

Runs all expensive computations (raw T=1 trajectories, tau sweeps,
temperature grid sweeps for all 5 methods) and saves to a pickle cache.
The dashboard loads from cache instead of recomputing on every page load.

Usage:
    python dashboard/precompute_adaptive.py [--n-max 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
import time
from pathlib import Path

import numpy as np
from scipy.special import betainc as _betainc, digamma as _digamma
from scipy.stats import chi2 as _chi2, gamma as _gamma, norm as _norm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_PATH = Path(__file__).resolve().parent / "adaptive_cache.pkl"
LOG_PATH = Path(__file__).resolve().parent / "precompute_adaptive.log"

# File + console logging so we can tail the log while the script runs
_fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
_fh.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[_fh, logging.StreamHandler()],
)
log = logging.getLogger(__name__)


class _FlushHandler(logging.Handler):
    """Flushes all handlers after every log record so tail -f works."""
    def emit(self, record):
        for h in logging.root.handlers:
            h.flush()


logging.root.addHandler(_FlushHandler())

# ---------------------------------------------------------------------------
# Data loading (duplicated from app.py to keep script standalone)
# ---------------------------------------------------------------------------


def _extract_run_prefix(stem: str) -> str:
    parts = stem.split("_")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) if parts else stem


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


def _extract_config_head(path: Path) -> dict | None:
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


def _effective_mode(cfg: dict) -> str:
    if cfg.get("think", False):
        return "think"
    return "CoT" if cfg.get("prompt_mode", "direct") == "cot" else "direct"


def _short_model_name(cfg: dict) -> str:
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
    parts = [_short_model_name(cfg)]
    parts.append(_effective_mode(cfg))
    if cfg.get("shuffle_options", False):
        parts.append("shuffle")
    if cfg.get("use_paraphrases", False):
        parts.append("paraphrase")
    return " · ".join(parts)


def results_to_df(all_data: dict[str, dict]) -> list[dict]:
    """Convert loaded result dicts to flat row dicts (no pandas dependency)."""
    rows = []
    seen: set[tuple[str, str]] = set()
    for run_name, data in all_data.items():
        cfg = data.get("config", {})
        model = _short_model_name(cfg)
        think = cfg.get("think", False)
        prompt_mode = cfg.get("prompt_mode", "direct")
        shuffle = cfg.get("shuffle_options", True)
        paraphrase = cfg.get("use_paraphrases", False)
        mode = _effective_mode(cfg)

        for qr in data.get("question_results", []):
            if qr.get("skipped", False) or not qr.get("mean_probs"):
                continue
            qid = qr.get("question_id", "")
            dedup_key = (run_name, qid)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            query_log = qr.get("query_log", [])
            final_answer = qr.get("final_answer", 0)

            rows.append({
                "run_name": run_name,
                "model": model,
                "mode": mode,
                "shuffle": shuffle,
                "paraphrase": paraphrase,
                "question_id": qid,
                "correct_answer": qr.get("correct_answer"),
                "difficult": qr.get("difficult", False),
                "final_answer": final_answer,
                "is_correct": qr.get("correct"),
                "query_log": query_log,
            })
    return rows


# ---------------------------------------------------------------------------
# Math helpers (duplicated from app.py)
# ---------------------------------------------------------------------------


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
    df = K * (N - 1)
    if mu_var_sum < 1e-15:
        return _exceedance_copula(mu * 10.0)

    R_hat = var_sum / mu_var_sum
    grid = _BAYES_GRID
    R_true = 1.0 / (grid + 1.0)
    ratio = R_hat / np.maximum(R_true, 1e-15) * df
    log_lik = _chi2.logpdf(ratio, df) + np.log(df) - np.log(np.maximum(R_true, 1e-15))
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


# ---------------------------------------------------------------------------
# Trajectory computation (duplicated from app.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


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


def _grid_sweep_all(questions, method, temps, taus, n_q, esc1_run, esc2_run):
    grid = []
    for t in temps:
        if method == "mle":
            trajs_t = _batch_mle_trajectories(questions, temp=t)
        else:
            trajs_t = [
                _compute_trajectory(q["probs"], method, t)
                for q in questions
            ]
        for tau_g in taus:
            ns_g, corrects_g, caps_g = [], [], []
            esc1_c, esc2_c, esc_best = [], [], []
            for i, traj in enumerate(trajs_t):
                q = questions[i]
                n_stop_g, capped_g = len(traj), True
                answer_g = traj[-1][1]
                for n, (conf, argmax_a) in enumerate(traj):
                    if conf > tau_g:
                        answer_g, n_stop_g, capped_g = argmax_a, n + 1, False
                        break
                ok = (answer_g == q["correct"])
                ns_g.append(n_stop_g)
                corrects_g.append(ok)
                caps_g.append(capped_g)
                esc_ok = ok
                if capped_g:
                    alpha_base = np.ones(4)
                    for p in q["probs"]:
                        alpha_base += np.array(p)
                    a2 = _augmented_answer(q, alpha_base, "esc2")
                    a1 = _augmented_answer(q, alpha_base, "esc1")
                    if a2 is not None:
                        esc_ok = (a2 == q["correct"])
                    elif a1 is not None:
                        esc_ok = (a1 == q["correct"])
                    esc1_c.append((a1 == q["correct"]) if a1 is not None else ok)
                    esc2_c.append((a2 == q["correct"]) if a2 is not None else ok)
                else:
                    esc1_c.append(ok)
                    esc2_c.append(ok)
                esc_best.append(esc_ok)
            avg_n_g = float(np.mean(ns_g))
            cap_rate_g = sum(caps_g) / n_q
            grid.append({
                "T": float(t), "tau": tau_g,
                "avg_n": avg_n_g, "cap_rate": cap_rate_g,
                "acc": sum(corrects_g) / n_q,
                "acc_esc": sum(esc_best) / n_q,
                "esc1": sum(esc1_c) / n_q if esc1_run else None,
                "esc2": sum(esc2_c) / n_q if esc2_run else None,
            })
    return grid


# ---------------------------------------------------------------------------
# Main: load data, compute everything, save cache
# ---------------------------------------------------------------------------

METHODS = [("Product", "prod"), ("Sum", "sum"), ("Dirichlet MLE", "mle"),
           ("MoM", "mom"), ("MoM + Bayes", "mom_bayes")]
MKEY_TO_METHOD = {"prod": "product", "sum": "sum", "mle": "mle",
                  "mom": "mom", "mom_bayes": "mom_bayes"}


def build_questions_data(rows: list[dict], n_max: int) -> dict:
    """Build per-model question data and run metadata from flat rows."""
    # Build run_meta
    run_meta: dict[str, dict] = {}
    run_to_prefix: dict[str, str] = {}
    for row in rows:
        rn = row["run_name"]
        if rn not in run_to_prefix:
            run_to_prefix[rn] = rn
            run_meta[rn] = {
                "model": row["model"],
                "mode": row["mode"],
                "shuffle": row["shuffle"],
                "paraphrase": row["paraphrase"],
            }

    # Auto-discover per-model configs
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
        return {}

    # Index rows by run_name
    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run_name"], []).append(row)

    # Build answer/probs maps
    def _answer_map(rn):
        if not rn or rn not in by_run:
            return {}
        return {r["question_id"]: r["final_answer"] for r in by_run[rn]}

    def _probs_map(rn):
        if not rn or rn not in by_run:
            return {}
        out = {}
        for r in by_run[rn]:
            for q in r.get("query_log", []):
                cp = q.get("canonical_probs", [])
                if len(cp) == 4:
                    out[r["question_id"]] = cp
                    break
        return out

    model_data: dict[str, dict] = {}
    for model_name, cfg in models_found.items():
        base_rows = by_run.get(cfg["base"], [])
        if not base_rows:
            continue
        esc1_map = _answer_map(cfg["esc1"])
        esc2_map = _answer_map(cfg["esc2"])
        esc1_probs = _probs_map(cfg["esc1"])
        esc2_probs = _probs_map(cfg["esc2"])
        questions = []
        for row in base_rows:
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
    return model_data


def run_precompute(n_max: int = 10) -> dict:
    """Run all adaptive sampling computations and return cache dict."""
    t0 = time.time()

    # --- Load data ---
    log.info("Loading result files...")
    result_files = get_result_files()
    if not result_files:
        log.info("No result files found.")
        return {}

    all_data: dict[str, dict] = {}
    for fp in result_files:
        data = _load_json(fp)
        if data:
            prefix = _extract_run_prefix(fp.stem)
            all_data[prefix] = data
    log.info(f"  Loaded {len(all_data)} runs.")

    rows = results_to_df(all_data)
    log.info(f"  {len(rows)} question rows.")

    model_data = build_questions_data(rows, n_max)
    if not model_data:
        log.info("No models with shuffle direct runs found.")
        return {}

    model_names = list(model_data.keys())
    log.info(f"  Models: {', '.join(model_names)}")

    # --- Raw trajectories (T=1) ---
    log.info("\nComputing raw trajectories (T=1)...")
    for mn, mdata in model_data.items():
        qs = mdata["questions"]
        log.info(f"  {mn}: {len(qs)} questions...")
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
    tau_values = [0.00, 0.60, 0.70, 0.80, 0.90, 0.95]
    log.info("\nSweeping raw tau values...")
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

    # --- Temperature grid sweep (all methods) ---
    temps_grid = np.arange(1.0, 5.5, 0.5)
    taus_grid = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    log.info("\nTemperature grid sweep (this is the slow part)...")
    cal_grids: dict[tuple[str, str], list] = {}
    total_sweeps = len(METHODS) * len(model_data)
    done = 0
    for mn, mdata in model_data.items():
        qs, n_q_m = mdata["questions"], mdata["n_q"]
        for method_label, mkey in METHODS:
            done += 1
            t_sweep = time.time()
            log.info(f"  [{done}/{total_sweeps}] {mn} × {method_label} ({n_q_m} qs × {len(temps_grid)} temps × {len(taus_grid)} taus)...")
            method = MKEY_TO_METHOD[mkey]
            cal_grids[(mkey, mn)] = _grid_sweep_all(
                qs, method, temps_grid, taus_grid, n_q_m,
                mdata["esc1_run"], mdata["esc2_run"],
            )
            log.info(f"    done in {time.time() - t_sweep:.1f}s")

    # --- Compute calibrated tau sweeps at optimal T ---
    tau_values_cal = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    log.info("\nComputing calibrated tau sweeps at optimal T...")
    cal_opt: dict = {}
    cal_model_data: dict = {}

    # Find optimal T per (method, model) from Pareto
    for (mkey, mn), grid in cal_grids.items():
        # Compute Pareto with a default esc_cost (we store the grid, dashboard re-derives)
        pass

    # Store optimal T for multiple esc_cost values — but the grid is what matters.
    # Dashboard will derive Pareto + optimal T on the fly from the cached grid (cheap).

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.1f}s.")

    # --- Build cache ---
    # Strip trajectories from model_data to save space (they're large).
    # We keep: questions (with probs), n_q, baseline_acc, has_esc1/2, esc1/2_run,
    #          and raw tau sweep results (prod_results, sum_results, etc.)
    model_data_slim = {}
    for mn, mdata in model_data.items():
        slim = {
            "questions": mdata["questions"],
            "n_q": mdata["n_q"],
            "baseline_acc": mdata["baseline_acc"],
            "has_esc1": mdata["has_esc1"],
            "has_esc2": mdata["has_esc2"],
            "esc1_run": mdata["esc1_run"],
            "esc2_run": mdata["esc2_run"],
        }
        for _, mkey in METHODS:
            slim[f"{mkey}_results"] = mdata[f"{mkey}_results"]
        model_data_slim[mn] = slim

    cache = {
        "n_max": n_max,
        "model_names": model_names,
        "model_data": model_data_slim,
        "tau_values": tau_values,
        "tau_values_cal": tau_values_cal,
        "cal_grids": cal_grids,
        "temps_grid": list(temps_grid),
        "taus_grid": taus_grid,
        "computed_at": time.time(),
    }
    return cache


def main():
    parser = argparse.ArgumentParser(description="Precompute adaptive sampling results")
    parser.add_argument("--n-max", type=int, default=10, help="Max permutations (default: 10)")
    args = parser.parse_args()

    cache = run_precompute(n_max=args.n_max)
    if not cache:
        log.info("Nothing to cache.")
        return

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = CACHE_PATH.stat().st_size / (1024 * 1024)
    log.info(f"Cache saved to {CACHE_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
