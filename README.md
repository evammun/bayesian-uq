# Bayesian Uncertainty Quantification for LLMs via Paraphrase-Based Sampling

A black-box Bayesian framework that combines paraphrase-based input diversification with sequential adaptive stopping to produce calibrated uncertainty estimates from LLM outputs — using only text-in, text-out API access.

## Setup

```bash
# Install uv (if not already installed)
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac: curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
uv add numpy scipy requests jupyter

# Activate environment
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

## Project Structure

```
src/bayesian_uq/     Core library (posterior updates, paraphrasing, querying, stopping)
experiments/          Experiment scripts
data/                 Datasets (MMLU, broken-premise pairs, paraphrases)
notebooks/            Exploratory analysis
results/              Experiment outputs (git-ignored)
paper/                Manuscript drafts
```

## Data

- **MMLU**: Download from [Hugging Face](https://huggingface.co/datasets/cais/mmlu) or use the test set in `data/mmlu/`
- **Broken-Premise Paired Dataset**: Original contribution — matched pairs of well-formed and broken-premise questions. See `data/broken_pairs/`
