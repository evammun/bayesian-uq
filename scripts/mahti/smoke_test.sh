#!/bin/bash
# ============================================================================
# Quick smoke test on Mahti gputest partition (15 min max, fast scheduling)
# ============================================================================
# Usage: sbatch scripts/mahti/smoke_test.sh
# ============================================================================

#SBATCH --job-name=uq-smoke
#SBATCH --account=project_2018384
#SBATCH --partition=gputest
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --output=/scratch/project_2018384/logs/%x_%j.out
#SBATCH --error=/scratch/project_2018384/logs/%x_%j.err

source /appl/profile/zz-csc-env.sh
module load gcc cuda python-data
source /projappl/project_2018384/llama-env/bin/activate

PROJECT=/scratch/project_2018384/bayesian-uq
cd "$PROJECT" || exit 1

# Copy model to NVMe
cp /scratch/project_2018384/models/qwen3-8b-q4_k_m.gguf $LOCAL_SCRATCH/
export UQ_MODEL_PATH=$LOCAL_SCRATCH/qwen3-8b-q4_k_m.gguf

echo "============================================"
echo "  Mahti Smoke Test"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================"
echo ""

srun python3 -c "
import sys, math, time
sys.path.insert(0, 'src')
from pre_action_uq.inference import LlamaCppClient, extract_answer_logprobs
from pre_action_uq.pipeline import load_quality_dataset, build_prompt, generate_permutation
from pathlib import Path
import os, random

model = os.environ['UQ_MODEL_PATH']

# --- Test 1: Load model ---
print('Test 1: Loading model (n_ctx=32768)...')
client = LlamaCppClient(model, n_ctx=32768, verbose=True)
print(f'  Loaded in {client.load_time:.1f}s')
print()

# --- Test 2: Simple MCQ ---
print('Test 2: Simple MCQ...')
r = client.generate_with_logprobs('What is 2+2?\n\nA) 3\nB) 4\nC) 5\nD) 6\n\nAnswer:', think=False)
print(f'  Answer: {r[\"response_text\"]} (should be B)')
assert 'B' in r['response_text'], f'Wrong answer: {r[\"response_text\"]}'
print('  PASS')
print()

# --- Test 3: Real QuALITY question (long article) ---
print('Test 3: Real QuALITY question...')
questions = load_quality_dataset(Path('data/quality_all.jsonl'))
rng = random.Random(42)
q = max(questions[:200], key=lambda x: len(x.article_text))
perm = generate_permutation(4, rng)
prompt = build_prompt(q.question_text, q.options, q.article_text, perm)
tokens = client.model.tokenize(prompt.encode(), add_bos=True)
print(f'  Article: {len(q.article_text.split())} words, {len(tokens)} tokens')
t0 = time.time()
r = client.generate_with_logprobs(prompt, think=False)
elapsed = time.time() - t0
disp_lp, _, disp_ans, canon_ans, _ = extract_answer_logprobs(r['logprobs'], perm)
correct = canon_ans == q.correct_answer
probs = ' '.join(f'{l}:{math.exp(disp_lp.get(l,-30)):.3f}' for l in 'ABCD')
print(f'  Answer: {disp_ans} | {probs} | {\"CORRECT\" if correct else \"WRONG\"} | {elapsed:.2f}s')
print('  PASS')
print()

# --- Test 4: Two-pass CoT ---
print('Test 4: CoT two-pass...')
cot_prompt = (
    'Read the following passage and answer the question.\n\n'
    f'Passage:\n{q.article_text}\n\n'
    f'Question: {q.question_text}\n\n'
    'A) ' + q.options[perm[0]] + '\n'
    'B) ' + q.options[perm[1]] + '\n'
    'C) ' + q.options[perm[2]] + '\n'
    'D) ' + q.options[perm[3]] + '\n\n'
    'BE CONCISE. Reason briefly, then end with:\nAnswer: X'
)
t0 = time.time()
r = client.generate_cot(prompt=cot_prompt, max_tokens=1024, temperature=0.7, think=False)
elapsed = time.time() - t0
print(f'  Pass 1 answer: \"{r.get(\"pass1_answer\", \"\")}\"')
print(f'  Reasoning (first 200): {r[\"thinking_trace\"][:200]}')
if r['logprobs']:
    disp_lp, _, disp_ans, _, _ = extract_answer_logprobs(r['logprobs'], perm)
    print(f'  Pass 2 answer: {disp_ans}')
print(f'  Time: {elapsed:.2f}s')
print('  PASS')
print()

# --- Test 5: KV cache reset (2 questions back-to-back) ---
print('Test 5: KV cache reset...')
r1 = client.generate_with_logprobs('What is the capital of France?\n\nA) London\nB) Paris\nC) Berlin\nD) Madrid\n\nAnswer:', think=False)
r2 = client.generate_with_logprobs('What is the capital of Japan?\n\nA) Beijing\nB) Seoul\nC) Tokyo\nD) Bangkok\n\nAnswer:', think=False)
t1 = r1['logprobs'][0]['top_logprobs'][0]['token'].strip()
t2 = r2['logprobs'][0]['top_logprobs'][0]['token'].strip()
print(f'  France: {t1} (should be B)')
print(f'  Japan: {t2} (should be C)')
assert t1 == 'B' and t2 == 'C', f'KV cache issue: got {t1}, {t2}'
print('  PASS')
print()

print('============================================')
print(f'ALL SMOKE TESTS PASSED ({client.query_count} queries)')
print('============================================')
"
