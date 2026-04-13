"""
Dump dashboard-equivalent summary to a text file that Claude can read.
Computes all signal battery metrics directly from result JSONs.

Usage: python experiments/dump_dashboard_summary.py
Output: results/dashboard_summary.txt
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT = RESULTS_DIR / "dashboard_summary.txt"

ANSWER_TOKENS = {" A", " B", " C", " D", "A", "B", "C", "D"}


def load_run(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        results = data.get("question_results", [])
        if not results:
            return None
        mode = "think" if cfg.get("think") else ("CoT" if cfg.get("prompt_mode") == "cot" else "direct")
        shuffle = cfg.get("shuffle_options", False)
        para = cfg.get("use_paraphrases", False)
        label = f"{mode} {'shuffle' if shuffle else 'noshuffle'}"
        if para:
            label += " para"
        return {"cfg": cfg, "results": results, "mode": mode, "shuffle": shuffle, "para": para, "label": label, "path": path}
    except Exception:
        return None


def compute_metrics(run: dict) -> dict:
    results = run["results"]
    valid = [r for r in results if r.get("correct") is not None]
    correct = [r for r in valid if r["correct"]]
    incorrect = [r for r in valid if not r["correct"]]

    def mean_or(vals, default=None):
        return sum(vals) / len(vals) if vals else default

    # Accuracy
    acc = len(correct) / len(valid) if valid else None

    # MSP
    msp_c = mean_or([max(r["mean_probs"]) for r in correct if r.get("mean_probs")])
    msp_i = mean_or([max(r["mean_probs"]) for r in incorrect if r.get("mean_probs")])

    # 2nd Gap
    def gap(r):
        mp = sorted(r["mean_probs"], reverse=True)
        return mp[0] - mp[1] if len(mp) >= 2 else None
    gap_c = mean_or([g for g in (gap(r) for r in correct) if g is not None])
    gap_i = mean_or([g for g in (gap(r) for r in incorrect) if g is not None])

    # Coverage
    def coverage(r):
        covs = []
        for q in r.get("query_log", []):
            tlps = []
            for entry in q.get("raw_logprobs", []):
                for t in entry.get("top_logprobs", []):
                    tlps.append((t.get("token", ""), t.get("logprob", -30)))
            if tlps:
                total = sum(math.exp(lp) for _, lp in tlps)
                ans = sum(math.exp(lp) for tok, lp in tlps if tok.strip() in ANSWER_TOKENS)
                if total > 0:
                    covs.append(ans / total)
        return mean_or(covs)
    cov_c = mean_or([c for c in (coverage(r) for r in correct) if c is not None])
    cov_i = mean_or([c for c in (coverage(r) for r in incorrect) if c is not None])

    # Epistemic
    def epistemic(r):
        qp = np.array([q["canonical_probs"] for q in r.get("query_log", []) if len(q.get("canonical_probs", [])) == 4])
        if len(qp) < 2:
            return None
        qp = np.clip(qp, 1e-12, None)
        md = qp.mean(axis=0)
        return -float(np.sum(md * np.log(md))) - float(np.mean(-np.sum(qp * np.log(qp), axis=1)))
    ep_c = mean_or([e for e in (epistemic(r) for r in correct) if e is not None])
    ep_i = mean_or([e for e in (epistemic(r) for r in incorrect) if e is not None])

    # Conf Var
    def conf_var(r):
        pqc = [max(q.get("canonical_probs", [.25]*4)) for q in r.get("query_log", []) if len(q.get("canonical_probs", [])) == 4]
        return float(np.std(pqc)) if len(pqc) > 1 else None
    cv_c = mean_or([v for v in (conf_var(r) for r in correct) if v is not None])
    cv_i = mean_or([v for v in (conf_var(r) for r in incorrect) if v is not None])

    # Agreement
    def agreement(r):
        answers = [q["canonical_answer"] for q in r.get("query_log", [])]
        if not answers:
            return None
        return Counter(answers).most_common(1)[0][1] / len(answers)
    ag_c = mean_or([a for a in (agreement(r) for r in correct) if a is not None])
    ag_i = mean_or([a for a in (agreement(r) for r in incorrect) if a is not None])

    # Trace length
    def trace_len(r):
        traces = [len(q.get("thinking_trace", "")) for q in r.get("query_log", []) if q.get("thinking_trace")]
        return mean_or(traces)
    tl_c = mean_or([t for t in (trace_len(r) for r in correct) if t is not None])
    tl_i = mean_or([t for t in (trace_len(r) for r in incorrect) if t is not None])

    # P1/P2 disagreement
    def p1p2_rate(r):
        total = disagree = 0
        for q in r.get("query_log", []):
            p1 = q.get("pass1_canonical_answer", -1)
            p2 = q.get("canonical_answer", -1)
            if p1 >= 0:
                total += 1
                if p1 != p2:
                    disagree += 1
        return disagree / total if total > 0 else None
    p1p2_c = mean_or([p for p in (p1p2_rate(r) for r in correct) if p is not None])
    p1p2_i = mean_or([p for p in (p1p2_rate(r) for r in incorrect) if p is not None])

    def delta(c, i):
        if c is not None and i is not None:
            return i - c
        return None

    def fmt(v, dp=2):
        return f"{v:.{dp}f}" if v is not None else "-"

    def fmtd(v, dp=2):
        return f"{v:+.{dp}f}" if v is not None else "-"

    return {
        "n": len(valid),
        "acc": f"{acc:.1%}" if acc is not None else "-",
        "msp_c": fmt(msp_c), "msp_d": fmtd(delta(msp_c, msp_i)),
        "gap_c": fmt(gap_c), "gap_d": fmtd(delta(gap_c, gap_i)),
        "cov_c": fmt(cov_c), "cov_d": fmtd(delta(cov_c, cov_i)),
        "ep_c": fmt(ep_c), "ep_d": fmtd(delta(ep_c, ep_i)),
        "cv_c": fmt(cv_c), "cv_d": fmtd(delta(cv_c, cv_i)),
        "ag_c": fmt(ag_c), "ag_d": fmtd(delta(ag_c, ag_i)),
        "tl_c": fmt(tl_c, 0), "tl_d": fmtd(delta(tl_c, tl_i), 0),
        "p1p2_c": fmt(p1p2_c, 3), "p1p2_d": fmtd(delta(p1p2_c, p1p2_i), 3),
    }


def main():
    # Load all runs
    runs = []
    for f in sorted(RESULTS_DIR.glob("quality_*.json")):
        if "test_" in f.name or "insufficient" in f.name:
            continue
        run = load_run(f)
        if run:
            runs.append(run)

    lines = [f"Dashboard Summary — {len(runs)} runs\n"]

    # Per-run metrics
    lines.append(f"{'Run':<40s} {'N':>5s} {'Acc':>6s}  {'MSP(c)':>6s} {'MSPd':>6s}  {'Gap(c)':>6s} {'Gapd':>6s}  {'Cov(c)':>6s} {'Covd':>6s}  {'Ep(c)':>6s} {'Epd':>6s}  {'CV(c)':>6s} {'CVd':>6s}  {'Ag(c)':>6s} {'Agd':>6s}  {'TL(c)':>6s} {'TLd':>6s}  {'P12(c)':>7s} {'P12d':>7s}")
    lines.append("-" * 180)

    for run in sorted(runs, key=lambda r: r["label"]):
        m = compute_metrics(run)
        lines.append(
            f"{run['label']:<40s} {m['n']:>5d} {m['acc']:>6s}  "
            f"{m['msp_c']:>6s} {m['msp_d']:>6s}  "
            f"{m['gap_c']:>6s} {m['gap_d']:>6s}  "
            f"{m['cov_c']:>6s} {m['cov_d']:>6s}  "
            f"{m['ep_c']:>6s} {m['ep_d']:>6s}  "
            f"{m['cv_c']:>6s} {m['cv_d']:>6s}  "
            f"{m['ag_c']:>6s} {m['ag_d']:>6s}  "
            f"{m['tl_c']:>6s} {m['tl_d']:>6s}  "
            f"{m['p1p2_c']:>7s} {m['p1p2_d']:>7s}"
        )

    # Cross-mode disagreement
    lines.append("")
    lines.append("Cross-Mode Disagreement (noshuffle, no-para runs)")
    lines.append(f"{'Comparison':<25s} {'N':>5s} {'Agree':>12s} {'Disagree':>12s} {'Acc(agree)':>10s} {'Acc(dis)':>10s}")

    noshuffle = {r["mode"]: {qr["question_id"]: qr for qr in r["results"]} for r in runs if not r["shuffle"] and not r["para"]}
    for a, b in [("CoT", "direct"), ("think", "direct"), ("think", "CoT")]:
        if a not in noshuffle or b not in noshuffle:
            continue
        common = set(noshuffle[a].keys()) & set(noshuffle[b].keys())
        agree_acc, disagree_acc = [], []
        for qid in common:
            ra, rb = noshuffle[a][qid], noshuffle[b][qid]
            same = ra.get("final_answer") == rb.get("final_answer")
            (agree_acc if same else disagree_acc).append(ra.get("correct", False))
        ag = len(agree_acc)
        dg = len(disagree_acc)
        total = ag + dg
        aa = sum(agree_acc) / ag if ag else 0
        da = sum(disagree_acc) / dg if dg else 0
        lines.append(f"{a+' vs '+b:<25s} {total:>5d} {ag:>5d} ({ag/total:.0%})   {dg:>5d} ({dg/total:.0%})   {aa:>9.1%} {da:>9.1%}")

    text = "\n".join(lines)
    OUTPUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
