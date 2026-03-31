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

## Pitfalls — Actual Issues Hit (2026-03-31)

These are real problems we hit during the first Mahti deployment, in order:

### 1. SSH key auth + MAC corruption (SOLVED)
- CSC uses SSH-key-only auth (added via MyCSC portal, takes up to 1 hour to propagate)
- Windows OpenSSH has a MAC algorithm incompatibility with Mahti: `Corrupted MAC on input`
- **Fix:** Add to `~/.ssh/config`:
  ```
  Host mahti.csc.fi
      MACs hmac-sha2-256
  ```
- **Pitfall:** Do NOT use PowerShell `echo >>` to edit SSH config — it writes UTF-16 with spaces between characters. Use a text editor or Claude Code's Write tool.

### 2. `module` command not found in SLURM jobs (SOLVED)
- The Lmod module system isn't initialized in non-interactive shells or SLURM batch scripts
- `module load gcc cuda python-data` silently does nothing, leaving system Python 3.6 in PATH
- **Fix:** Add `source /appl/profile/zz-csc-env.sh` as the FIRST line after SBATCH directives, before any `module load`
- This is the CSC-specific init that sets up MODULEPATH and the module function

### 3. `srun` doesn't inherit environment (SOLVED)
- In SLURM scripts, `srun python3` starts a subprocess that doesn't see the venv activation or loaded modules
- **Fix:** Don't use `srun` — just call `python3` directly. The batch script's environment is inherited by child processes without `srun`.

### 4. llama-cpp-python installs CPU-only by default (SOLVED)
- `pip install llama-cpp-python` from the default index gives a CPU-only wheel
- The CUDA wheel index (`whl/cu124`) only has wheels for CUDA 12.x; Mahti has CUDA 11.5
- `llama_cpp.llama_supports_gpu_offload()` returns `False` silently — the pipeline runs but 100x slower
- **Fix:** Build from source with `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80"`
- **BUT:** This fails on the login node because `libcuda.so` (the CUDA driver) only exists on GPU compute nodes
- **Fix for the fix:** Submit the build as a SLURM job on `gputest` partition:
  ```bash
  sbatch --partition=gputest --time=0:15:00 --gres=gpu:a100:1,nvme:100 --wrap='bash /scratch/.../build_llama.sh'
  ```
- Build takes ~8 minutes on compute node (377 compilation units, sm_80 only)
- **Always verify after install:** `python3 -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"` must print `True`

### 5. YAML files with Windows encoding (SOLVED)
- Python's `write_text()` on Windows can embed em-dash characters (U+2014) as single byte 0x97 (Windows-1252) instead of the 3-byte UTF-8 sequence
- YAML parser on Linux chokes: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x97`
- **Fix:** Write YAML files as pure ASCII — no special characters in comments. Use `write_text(content, encoding='ascii')` to catch any non-ASCII chars at write time
- **Also:** `sed -i 's/\r$//'` after every scp to strip CRLF (same issue as vast.ai)

### 6. HuggingFace model URL case sensitivity (SOLVED)
- The GGUF filename on HuggingFace is `Qwen3-8B-Q4_K_M.gguf` (capital letters)
- Our code references `qwen3-8b-q4_k_m.gguf` (lowercase)
- wget with the wrong case returns 404
- **Fix:** Download with correct URL, save as lowercase: `wget -O qwen3-8b-q4_k_m.gguf https://...Qwen3-8B-Q4_K_M.gguf`

### 7. CoT wall time might be tight (MONITORING)
- CoT noshuffle: ~6s per question x 4609 = ~7.7 hours. Wall time set to 6 hours.
- May need resume after wall-time kill, or increase to `--time=10:00:00`
- Resume logic handles this automatically

---

## Verified Working Configuration (2026-03-31)

```
Cluster:        Mahti (mahti.csc.fi)
Node:           g2102.mahti.csc.fi
GPU:            NVIDIA A100-SXM4-40GB
CUDA:           11.5.0 (module)
CUDA driver:    535.288.01 (on compute nodes)
GCC:            11.2.0 (Spack)
Python:         3.12.11 (python-data module)
llama-cpp-python: 0.3.19 (built from source, GPU=True)
n_ctx:          32768
Model:          qwen3-8b-q4_k_m.gguf (4.7 GB)
Load time:      4.4s
Direct query:   ~2.5s per question
CoT query:      ~6s per question
```

**Init sequence for SLURM scripts:**
```bash
source /appl/profile/zz-csc-env.sh   # MUST be first — sets up module system
module load gcc cuda python-data
source /projappl/project_2018384/llama-env/bin/activate
export UQ_MODEL_PATH=$LOCAL_SCRATCH/qwen3-8b-q4_k_m.gguf
```

---

## Budget Planning (updated with actual timings)

| Experiment | Queries/q | Est. hours (A100) | BU cost |
|-----------|-----------|-------------------|---------|
| direct x noshuffle x 2 contexts | 1 | ~1.5h each | 300 |
| direct x shuffle x 2 contexts | 10 | ~7h each | 1,400 |
| cot x noshuffle x 2 contexts | 1 | ~8h each | 1,600 |
| cot x shuffle x 2 contexts | 10 | ~80h each | 16,000 |
| **Total** | | **~195h** | **~19,300** |

CoT is slower than initially estimated (~6s/q vs ~3s predicted). Shuffle+CoT experiments are very expensive.

---

## Code changes required

**Core inference code (`src/pre_action_uq/`):** NONE. Works as-is on Mahti without modification.

**Mahti-specific scripts (`scripts/mahti/`):**
- `slurm_single.sh` — SLURM batch script (sources zz-csc-env.sh, no srun)
- `submit_all.sh` — submit all 8 experiments with tailored wall times
- `setup_env.sh` — one-time environment setup
- `smoke_test.sh` — 5-test validation on gputest partition
- `fetch_results.ps1` — pull results from Mahti to local

**Mahti configs (`experiments/configs/mahti/`):**
- All 8 experiments with n_ctx=32768 (vs 12288 on vast.ai)
- Pure ASCII encoding (no em-dashes or special chars)

**vast.ai scripts (`scripts/`) unchanged** — different execution model, same core code.
