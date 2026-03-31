# CSC Mahti Setup — Specification & Migration Roadmap

## What we're doing

Running LLM inference for an academic UQ research project. Extract logprobs from multiple-choice reading comprehension questions (QuALITY dataset) using llama-cpp-python with a GGUF model on an NVIDIA A100 40GB GPU.

**Not training. Not fine-tuning. Single-GPU batch inference only.**

## CSC specifics

- **Cluster:** Mahti (`mahti.csc.fi`)
- **Project:** `project_2018384`
- **Account for SLURM:** `--account=project_2018384`
- **GPU partition:** `gpusmall` (1-2 GPUs per node, 36h max wall time)
- **Test partition:** `gputest` (15 min, fast scheduling — use for debugging)
- **GPU type:** A100 40GB, request with `--gres=gpu:a100:1`
- **Billing:** 100 BU/hour per A100. CPU cores and memory are NOT separately billed on GPU partitions (unlike Puhti). NVMe is negligible (~0.01 BU/GiB/hour).
- **Module system:** Lmod. Key modules: `gcc`, `cuda`, `python-data`
- **SSH auth:** SSH keys only (added via MyCSC portal). Allow up to 1 hour after adding key for activation.

## Storage layout

| Path | Use for | Quota | Purge | Notes |
|------|---------|-------|-------|-------|
| `/users/<username>` (home) | SSH keys, tiny configs only | 10 GiB, 100K files | Never | Do NOT put models, venvs, or code here |
| `/projappl/project_2018384/` | Python venv, installed packages | 50 GiB, 100K files | Never | Persistent. Beware file count limit — venvs eat into it |
| `/scratch/project_2018384/` | GGUF model, code, data, results | 1 TiB, 1M files | 180 days no-access | Primary working area |
| `$LOCAL_SCRATCH` | Copy model here at job start | ~3.8 TiB NVMe | **Purged when job ends** | Fast I/O. Only exists during batch/interactive jobs |

**No backups exist on any CSC storage.** Manage your own backups via local download or Allas object storage.

## The inference pipeline

The Python script does this per question:

1. Build prompt using Qwen3 chat template with `Answer:` as **assistant prefill**
2. Tokenize full prompt
3. `model.eval(tokens)` with `logits_all=False` — forward pass, only last position's logits stored
4. `llama_cpp.llama_get_logits(model._ctx.ctx)` — raw pointer to last position's logits
5. Copy to numpy, log-softmax, extract top-20 + force-include A/B/C/D token IDs
6. Clear KV cache between questions (multi-version fallback: `llama_memory_seq_rm` → `memory_clear` → `kv_cache_clear`)
7. Save results incrementally to JSON (every 10 questions)

**Critical details:**
- `logits_all=False` essential — `True` allocates vocab×n_ctx×4 bytes (~7GB for Qwen3)
- `n_batch` must equal `n_ctx` (12288) for long prompts
- `model.reset()` does NOT clear KV cache — our `_full_reset()` handles this with version fallbacks
- Model path discovered via `UQ_MODEL_PATH` env var (set in SLURM script)

## Chat template

```
<|im_start|>system\n/no_think\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{passage+question+options}<|im_end|>\n<|im_start|>assistant\nAnswer:
```

---

# Migration Roadmap — vast.ai → CSC Mahti

## Architecture difference

| | vast.ai | CSC Mahti |
|---|---|---|
| Execution | Long-running Docker container | SLURM batch jobs with wall-time limits |
| Interruptions | Spot eviction, container persists on resume | Job dies on wall-time, re-submit to resume |
| Concurrency | One experiment at a time | Multiple jobs in parallel (if BUs allow) |
| Model location | `/workspace/models/` | Copy from `/scratch/` to `$LOCAL_SCRATCH` per job |
| Internet | Available always | **NOT on compute nodes** — login nodes only |
| Max runtime | Unlimited (until eviction) | 36 hours (gpusmall) |
| GPU | RTX 5090 (rented) | A100 40GB (allocated) |

**Key insight:** Instead of `run_all_pilots.sh` (one loop doing everything), each experiment becomes a separate SLURM job. This is better — jobs run independently, can be parallel, and wall-time limits are per-experiment.

---

## Phase 1: One-Time Setup

### 1.1 Upload code and data

From local machine (Git Bash or WSL):
```bash
rsync -azP \
  --exclude='.venv' --exclude='__pycache__' --exclude='results/' \
  --exclude='v2_mmlu_archive/' --exclude='.git' --exclude='*.pyc' \
  --exclude='Temp crappy folder random stuff/' \
  "/c/Users/evama/Dropbox/Family Room/Projects/bayesian-uq/" \
  username@mahti.csc.fi:/scratch/project_2018384/bayesian-uq/
```

Then fix Windows line endings:
```bash
ssh username@mahti.csc.fi
cd /scratch/project_2018384/bayesian-uq
# If dos2unix is available:
find . -name "*.sh" -o -name "*.yaml" | xargs dos2unix 2>/dev/null
# If not, use sed:
find . -name "*.sh" -o -name "*.yaml" -exec sed -i 's/\r$//' {} +
```

**Pitfall: CRLF.** Already burned us on vast.ai — the `\r` in YAML run_name broke all file matching. Must fix after every rsync from Windows.

### 1.2 Download model (login node)

Login nodes have internet. Compute nodes do NOT.

```bash
mkdir -p /scratch/project_2018384/models
cd /scratch/project_2018384/models
wget https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/qwen3-8b-q4_k_m.gguf
ls -lh qwen3-8b-q4_k_m.gguf  # should be ~5.2 GB
```

**Pitfall:** If download is interrupted, partial file will crash the pipeline with a GGUF parsing error. Verify size. Delete and re-download if wrong.

### 1.3 Build Python environment

The venv goes on `/projappl/` (persistent, not auto-cleaned):

```bash
# On login node (has internet for pip)
module load gcc cuda python-data
cd /projappl/project_2018384
python3 -m venv --system-site-packages llama-env
source llama-env/bin/activate
```

**`--system-site-packages` is critical** — gives access to CSC's optimised numpy/scipy/pandas. Do NOT pip-install these.

### 1.4 Install llama-cpp-python (highest risk step)

```bash
# Still on login node with venv activated and modules loaded

# Option A: Try pre-built CUDA wheel (fastest)
pip install "llama-cpp-python>=0.3.16" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# If Option A fails (likely due to glibc/CUDA mismatch), Option B: build from source
CMAKE_ARGS="-DGGML_CUDA=on" \
CUDA_ARCHITECTURES="80" \
CUDACXX=$(which nvcc) \
pip install llama-cpp-python --no-binary llama-cpp-python --verbose
```

**Pitfall: CUDA version.** Check `module avail cuda` and `nvcc --version`. The A100 needs compute capability 8.0. If CUDA module is 11.x, use `whl/cu118` instead of `whl/cu124`.

**Pitfall: GCC version.** llama.cpp needs GCC ≥ 11 for C++17. Check `gcc --version` after `module load gcc`. If too old, try `module load gcc/11.3.0` or similar.

**Pitfall: Compilation on login node.** CSC allows light compilation on login nodes. llama-cpp-python compilation takes 3-5 minutes and ~1 core — within policy. If nervous, use `sinteractive` but verify it has internet (it might not).

**Pitfall: File count limit.** `/projappl/` has 100K file limit. A large venv eats into this. After install, clean up: `find /projappl/project_2018384/llama-env -name '__pycache__' -exec rm -rf {} + 2>/dev/null`

### 1.5 Install remaining Python deps

```bash
pip install pydantic>=2.0 pyyaml orjson tqdm
# numpy, scipy, pandas, scikit-learn come from system-site-packages
```

### 1.6 Verify installation

```bash
python3 -c "
import llama_cpp; print(f'llama-cpp-python {llama_cpp.__version__}')
import pydantic; print(f'pydantic {pydantic.__version__}')
import numpy; print(f'numpy {numpy.__version__}')
import yaml; print('pyyaml OK')
import orjson; print('orjson OK')
print('ALL DEPS OK')
"
```

---

## Phase 2: Smoke Test (gputest partition — 15 min, fast scheduling)

### 2.1 Interactive GPU test

```bash
sinteractive --account project_2018384 --time 0:15:00 --gres=gpu:a100:1,nvme:100

module load gcc cuda python-data
source /projappl/project_2018384/llama-env/bin/activate
cd /scratch/project_2018384/bayesian-uq

# Copy model to fast NVMe
cp /scratch/project_2018384/models/qwen3-8b-q4_k_m.gguf $LOCAL_SCRATCH/
export UQ_MODEL_PATH=$LOCAL_SCRATCH/qwen3-8b-q4_k_m.gguf

python3 -c "
import sys; sys.path.insert(0, 'src')
from pre_action_uq.inference import LlamaCppClient
import os, math

model = os.environ['UQ_MODEL_PATH']
print('Loading model...')
client = LlamaCppClient(model, n_ctx=12288, verbose=True)
print(f'Loaded in {client.load_time:.1f}s')

# Test 1: simple MCQ
r1 = client.generate_with_logprobs('What is 2+2?\n\nA) 3\nB) 4\nC) 5\nD) 6\n\nAnswer:', think=False)
print(f'Test 1 — top token: {r1[\"response_text\"]}, n_logprobs: {len(r1[\"logprobs\"][0][\"top_logprobs\"])}')

# Test 2: long prompt (real QuALITY article)
from pre_action_uq.pipeline import load_quality_dataset, build_prompt, generate_permutation
from pathlib import Path
import random
questions = load_quality_dataset(Path('data/quality_all.jsonl'))
q = max(questions[:100], key=lambda x: len(x.article_text))  # longest article
perm = generate_permutation(4, random.Random(42))
prompt = build_prompt(q.question_text, q.options, q.article_text, perm)
tokens = client.model.tokenize(prompt.encode(), add_bos=True)
print(f'Test 2 — article words: {len(q.article_text.split())}, prompt tokens: {len(tokens)}')
r2 = client.generate_with_logprobs(prompt, think=False)
print(f'Test 2 — top token: {r2[\"response_text\"]}, OK')

# Test 3: verify KV cache reset works (run 2 questions back-to-back)
r3 = client.generate_with_logprobs('What is the capital of France?\n\nA) London\nB) Paris\nC) Berlin\nD) Madrid\n\nAnswer:', think=False)
for e in r3['logprobs'][0]['top_logprobs'][:4]:
    print(f'  {e[\"token\"]:>6s}: {math.exp(e[\"logprob\"]):.4f}')

print(f'\\nALL SMOKE TESTS PASSED (queries: {client.query_count})')
"
```

**What to verify:**
- [ ] Model loads, shows "offloaded 33/33 layers to GPU"
- [ ] Test 1: answer is "B" with high probability
- [ ] Test 2: long article (~8000 tokens) doesn't crash
- [ ] Test 3: logprobs are sensible (not garbage from stale KV cache)
- [ ] No CUDA errors, no segfaults

### 2.2 Batch test (gputest, 15 min)

Create a test config with `max_questions: 20`:
```bash
cat > /scratch/project_2018384/bayesian-uq/experiments/configs/test_mahti.yaml << 'EOF'
run_name: test_mahti_smoke
model_path: auto
think: false
prompt_mode: direct
dataset_file: data/quality_all.jsonl
context_condition: sufficient
shuffle_options: false
max_questions: 20
seed: 42
temperature: 0.7
num_permutations: 1
n_ctx: 12288
EOF
```

Submit:
```bash
sbatch scripts/mahti/slurm_single.sh experiments/configs/test_mahti.yaml
```

Check output:
```bash
squeue --me  # see job status
cat /scratch/project_2018384/logs/uq-*_<jobid>.out  # stdout
cat /scratch/project_2018384/logs/uq-*_<jobid>.err  # stderr
ls -lh results/test_mahti_smoke_*.json  # result file
```

---

## Phase 3: Full Experiment Runs

### 3.1 Submit all experiments

Use the submission script (`scripts/mahti/submit_all.sh`) which:
- Checks each config for completion status
- Sets wall time based on num_permutations (noshuffle=2h, shuffle=20h)
- Passes config path to the SLURM script via `--export`

```bash
cd /scratch/project_2018384/bayesian-uq
bash scripts/mahti/submit_all.sh
```

### 3.2 Monitor

```bash
squeue --me                        # job queue
sacct --format=JobID,Elapsed,State # completed jobs
tail -f /scratch/project_2018384/logs/uq-*.out  # live output
```

### 3.3 Wall-time resumption

Shuffle experiments (4609 × 10 queries) take ~15-20 hours on A100. This fits within the 36-hour gpusmall limit. But if a job gets killed (wall-time or preemption), re-submitting the same config will auto-resume from the partial result file.

---

## Phase 4: Retrieve Results

```bash
# From local machine
rsync -azP \
  username@mahti.csc.fi:/scratch/project_2018384/bayesian-uq/results/ \
  "/c/Users/evama/Dropbox/Family Room/Projects/bayesian-uq/results/"
```

Also back up to persistent storage:
```bash
# On Mahti
cp -r /scratch/project_2018384/bayesian-uq/results/ \
      /projappl/project_2018384/results_backup/
```

---

## Pitfalls — Ranked by Likelihood

### HIGH risk
1. **llama-cpp-python compilation.** CUDA version, GCC version, CMake flags. Try wheel first, source build second. Test explicitly before submitting real jobs.
2. **CRLF line endings.** Fix after every rsync from Windows. We already got burned on vast.ai.
3. **No internet on compute nodes.** ALL downloads must happen on login node. pip install on login node. Model download on login node.

### MEDIUM risk
4. **llama-cpp-python API version differences.** `memory_clear`, `llama_get_logits` APIs change between versions. Our `_full_reset` has fallbacks. Test with 2+ sequential questions to verify KV cache actually clears.
5. **Path resolution.** `dataset_file: data/quality_all.jsonl` is relative — resolved from project root via `Path(__file__)`. Works IF the SLURM script `cd`s to the project directory. Always `cd $PROJECT` before running.
6. **File count limit on `/projappl/`.** 100K files. Clean `__pycache__` after venv setup.

### LOW risk
7. **Scratch auto-cleanup.** 180 days. Set a calendar reminder.
8. **BU budget exhaustion.** Jobs get killed. Resume handles it.
9. **Wall-time on shuffle experiments.** 36h limit should suffice (~15-20h estimated). If not, resume.

---

## Budget Planning

| Experiment | Queries/q | Total queries | Est. hours (A100) | BU cost |
|-----------|-----------|---------------|-------------------|---------|
| direct × noshuffle × sufficient | 1 | 4,609 | ~1.5h | 150 |
| direct × noshuffle × insufficient | 1 | 4,609 | ~1.5h | 150 |
| direct × shuffle × sufficient | 10 | 46,090 | ~15h | 1,500 |
| direct × shuffle × insufficient | 10 | 46,090 | ~15h | 1,500 |
| cot × noshuffle × sufficient | 1 | 4,609 | ~4h | 400 |
| cot × noshuffle × insufficient | 1 | 4,609 | ~4h | 400 |
| cot × shuffle × sufficient | 10 | 46,090 | ~40h | 4,000 |
| cot × shuffle × insufficient | 10 | 46,090 | ~40h | 4,000 |
| **Total** | | **~199,000** | **~121h** | **~12,100** |

Current allocation: 1,000 BU = enough for the 2 noshuffle direct experiments as validation.

**Strategy:** Run noshuffle experiments first (~300 BU). Validate results match vast.ai. Then request medium allocation (~15,000 BU) for the full set. Run direct shuffle next (~3,000 BU). CoT experiments last (~8,800 BU) — they're the most expensive and may not be needed depending on direct-mode results.

---

## Code changes required

**Core inference code (`src/pre_action_uq/`):** NONE. Works as-is. `UQ_MODEL_PATH` env var handles model discovery. All paths relative to project root.

**New scripts needed (`scripts/mahti/`):**
- `slurm_single.sh` — SLURM batch script for one experiment
- `submit_all.sh` — loop to submit all 8 experiments
- `setup_env.sh` — one-time environment setup helper

**Existing scripts unchanged:**
- `scripts/run_all_pilots.sh` — vast.ai only
- `scripts/autorun.sh` — vast.ai only
- `scripts/vastai_setup.sh` — vast.ai only
- `scripts/fetch_results.ps1` — update SSH target for Mahti downloads

**Config changes:** None. All 8 YAML configs work as-is. `model_path: auto` uses `UQ_MODEL_PATH`.
