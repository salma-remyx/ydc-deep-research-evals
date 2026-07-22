# DeepConsult: A Deep Research Benchmark for Consulting / Business Queries

This repository contains evals metrics and scripts for evaluating Deep Research reports.

## Overview
We provide a pairwise-comparison evaluation script that was inspired by Google's Deep Research capabilities as described in their [blog post about Deep Research on Gemini 2.5 Pro Experimental](https://blog.google/products/gemini/deep-research-gemini-2-5-pro-experimental/).

The evaluation is performed by comparing generated research reports against reference reports (in our case OpenAI's Deep Research outputs) across four key dimensions:

1. **Instruction Following**: Evaluates response's fidelity to user specified instructions and constraints.
2. **Comprehensiveness**: Measures breadth and range of information covered in response, addressing the scope of user request.
3. **Completeness**: Measures the depth and thoroughness of information for topics addressed in the report.
4. **Writing Quality**: Evaluates clarity, conciseness, logical organization, and overall readability of the report.

These dimensions align with the capabilities that make Deep Research tools effective at analytical reasoning, information synthesis, and generating insightful research reports.


## DeepConsult Dataset

We include the DeepConsult dataset in the `datasets/DeepConsult` directory, which consists of:

1. `queries.csv` - A collection of business and consulting-related prompts designed for deep research. These queries cover a wide range of topics including:
   - Market analysis and investment opportunities
   - Industry-specific evaluations
   - Financial modeling and assessment
   - Technology trend analysis
   - Strategic business planning

2. `responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv` - This file contains responses from OpenAI DeepResearch and ARI formatted specifically for use with the evaluation script. The file follows the required format for the evals script:
   - `question`: The original research questions/prompts
   - `baseline_answer`: Responses from OpenAI's Deep Research capabilities (used as reference)
   - `candidate_answer`: Responses from ARI to be evaluated against the baseline
   
The dataset is designed to benchmark and evaluate the capability of language models to perform deep research on complex business and consulting queries, assessing their ability to provide comprehensive, well-structured, and insightful analysis comparable to professional consulting reports.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-org/ydc-deep-research-evals.git
   cd ydc-deep-research-evals
   ```

2. Install Git LFS (Large File Storage) if you don't have it already:  
(This is required for downloading the example dataset in this repo)
   ```bash
   # On Ubuntu/Debian
   apt-get install git-lfs
   
   # On macOS (using Homebrew)
   brew install git-lfs
   
   # Initialize Git LFS
   git lfs install
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables for OpenAI API access:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export OPENAI_ORGANIZATION_ID=your_openai_org_id
   ```

## Usage

### Running Evaluations

To evaluate research-style responses, use the `deep_research_pairwise_evals.py` script:

```bash
python evals/deep_research_pairwise_evals.py \
  --input-data datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv \
  --output-dir path/to/output/directory \
  --model o3-mini-2025-01-31 \
  --num-workers 4 \
  --metric-num-workers 3 \
  --metric-num-trials 3
```

### Input Data Format

The input CSV file should contain the following columns:
- `question`: The research question or prompt
- `baseline_answer`: The reference answer to compare against
- `candidate_answer`: The candidate answer to evaluate

### Output

The evaluation results are saved as a JSONL file in the specified output directory. Each line contains the evaluation results for a single question-answer pair, including:

- Original input data (question, answers, metadata)
- Scores for each evaluation dimension
- Aggregate metrics
- Raw evaluation data

## Using the Evaluation Metric in Your Code

You can also use the evaluation metric directly in your Python code:

```python
from evals.metrics.deep_research_pairwise_metric import DeepResearchPairwiseMetric

# Initialize the metric
metric = DeepResearchPairwiseMetric(
    eval_model="o3-mini-2025-01-31",
    num_trials=3,
    num_workers=3
)

# Evaluate a single question-answer pair
result = metric.score(
    question="What are the impacts of climate change on agriculture?",
    baseline_answer="Your reference answer text...",
    candidate_answer="Your candidate answer text..."
)

# Access the evaluation results
print(f"Instruction Following Score: {result.instruction_following.score}")
print(f"Comprehensiveness Score: {result.comprehensiveness.score}")
print(f"Completeness Score: {result.completeness.score}")
print(f"Writing Quality Score: {result.writing_quality.score}")
```

## Configuration

The evaluator supports several configuration options:

- `model`: The OpenAI model to use for evaluation (default: o3-mini-2025-01-31)
- `num-workers`: Number of worker threads for parallel processing (default: 4)
- `metric-num-workers`: Number of worker threads for the underlying metric (default: 3)
- `metric-num-trials`: Number of trials per evaluation for more stable results (default: 3)
  - For each trial, the evaluation runs twice - once with the original order and once with the baseline and candidate answers flipped. This helps mitigate potential position bias in the evaluation.

## Multi-Agent Debate Judge

An alternative **judge architecture** that scores the same report pairs via multi-agent debate rather than a single model pass. A panel of debating agents independently assesses the pair from distinct perspectives, revises their verdicts over a number of debate rounds in light of one another's evidence, and a final chair agent synthesizes the consolidated verdict — producing the same per-dimension preference as the single-pass judge, so the two architectures' results are directly comparable.

This lets you test whether debate beats the single-pass judge on the DeepConsult dataset — adapted from [Does Multi-Agent Debate Improve AI Feedback on Research Papers?](https://arxiv.org/abs/2607.14713v1), whose headline finding is that authors preferred the single pass over multi-agent debate tools despite the latter spending roughly thirty times the tokens. Run it alongside the single-pass evaluator (its outputs use a `debate_` prefix so the two coexist):

```bash
python -m evals.multi_agent_debate_evals \
  --input-data datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv \
  --output-dir path/to/output/directory \
  --model o3-mini-2025-01-31 \
  --debaters 3 \
  --debate-rounds 1 \
  --metric-num-trials 3
```

You can also use the debate metric directly in your code, where it subclasses the standard pairwise metric and reuses its aggregation:

```python
from evals.multi_agent_debate_metric import MultiAgentDebateMetric

metric = MultiAgentDebateMetric(
    eval_model="o3-mini-2025-01-31",
    num_trials=3,
    n_debaters=3,
    n_rounds=1,
)
result = metric.score(
    question="What are the impacts of climate change on agriculture?",
    baseline_answer="Your reference answer text...",
    candidate_answer="Your candidate answer text...",
)
```

## License

This project is licensed under the terms included in the LICENSE file.

## Requirements

- Python 3.10+
- OpenAI API access
- Dependencies listed in requirements.txt
