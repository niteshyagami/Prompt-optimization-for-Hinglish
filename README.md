# Comparative Analysis of Prompt Engineering vs. Context Engineering Techniques for Hindi/Hinglish Student Question Answering in Resource-Constrained Environments

## Overview

This project investigates which **prompt engineering** and **context engineering** strategies deliver the best quality-cost tradeoff for Hindi/Hinglish student question answering systems operating under resource constraints. The study compares multiple prompting techniques across different LLM sizes to identify optimal strategies for low-resource settings.

## Problem Statement

Large Language Models (LLMs) exhibit significant performance variation across different prompt designs. In resource-constrained environments (small/fast models, token budgets, latency requirements), identifying the optimal balance between answer quality and computational cost is critical. This study systematically evaluates multiple prompting and lightweight context strategies on a Hinglish QA dataset.

## Key Definitions

- **Prompt Engineering**: Deliberate design of instructions, examples, and formatting to steer model behavior without external knowledge sources (zero-shot, few-shot, chain-of-thought, etc.)

- **Context Engineering (Lightweight)**: Prompt structures that add reasoning frameworks or pseudo-context without external retrieval or long documents (hierarchical context, tree-of-thought, agentic steps, etc.)

- **Resource-Constrained Environment**: Systems limited by small/fast models, token budgets, and latency requirements with emphasis on low inference costs

## Features

- 🎯 **8 Prompting Strategies Tested**: Zero-shot, Few-shot, Chain-of-Thought, Structured Context, Hierarchical Context, RAG-style, Tree-of-Thought, Agentic Context
- 🌐 **Hinglish Support**: Roman-script Hindi/Hinglish (natural Indian English-Hindi mix)
- 📊 **Comparative Analysis**: Two LLM sizes (8B and 70B parameters)
- 📈 **Statistical Validation**: ANOVA, Tukey HSD, and variance analysis
- 💰 **Cost Efficiency Metrics**: Token usage and latency analysis
- 🧠 **Human Evaluation**: 5-annotator consensus scoring

## Project Structure

### Core Files

```
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── prompt_templates.py                          # All 8 prompting strategies
│
├── DATASET CREATION PIPELINE
├── load_dataset.py                              # Load Kaggle JSONL data
├── convert_to_hinglish.py                       # Convert to Hinglish via Groq
├── generate_new_samples.py                      # Generate additional samples
├── merge_datasets.py                            # Merge and prepare final dataset
├── final_hinglish_qa_dataset.csv                # Final merged dataset
│
├── EVALUATION PIPELINE
├── fast_test_experiment.py                      # Quick baseline run (50 questions)
├── medium_test_experiment.py                    # Full run (200 questions)
├── evaluate_results.py                          # Score answers using GPT-4
├── create_human_eval_template.py                # Prepare human evaluation
├── generate_human_responses.py                  # Generate human answer set
│
├── ANALYSIS & VISUALIZATION
├── analyze_human_vs_llm.py                      # Compare human vs LLM quality
├── generate_error_analysis.py                   # Categorize error types
├── generate_variance_table.py                   # Compute variance metrics
├── run_stats_tests.py                           # ANOVA and Tukey HSD tests
├── generate_all_paper_figures.py                # Generate publication figures
├── create_main_results_table.py                 # Consolidated results table
├── regenerate_all_tables.py                     # Regenerate all outputs
│
└── RESULTS & DATA FILES (see Output Files section below)
```

## Installation

### Prerequisites
- Python 3.8+
- Groq API key (for dataset generation)
- OpenAI API key (for evaluation)

### Setup

```bash
# Clone or navigate to project directory
cd "Comparative Analysis of Prompt Engineering vs. Context Engineering Techniques for HindiHinglish Student Question Answering in Resource-Constrained Environments - Copy"

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY="your_groq_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

## Dataset Pipeline

### Stage 1: Data Loading & Cleaning
```bash
python load_dataset.py
# Output: student_qa_clean.csv (cleaned Kaggle educational QA data)
```

### Stage 2: English to Hinglish Conversion
```bash
python convert_to_hinglish.py
# Output: hinglish_student_qa_final.csv (Hinglish versions via Groq translation)
```

### Stage 3: Sample Generation
```bash
python generate_new_samples.py
# Output: new_hinglish_samples_200.csv (200 synthetically generated samples)
```

### Stage 4: Dataset Merging
```bash
python merge_datasets.py
# Output: final_hinglish_qa_dataset.csv (merged + shuffled final dataset)
```

## Experimental Setup

### Prompting Strategies Tested

1. **Zero-shot**: Direct question with minimal instruction
2. **Few-shot**: 2-3 example Q&A pairs before the question
3. **Chain-of-Thought**: Explicit step-by-step reasoning instructions
4. **Structured Context**: JSON/XML formatted output templates
5. **Hierarchical Context**: Multi-level reasoning structure
6. **RAG-style**: Simulated retrieval-augmented generation (prompt-only context)
7. **Tree-of-Thought**: Multiple reasoning paths with convergence
8. **Agentic Context**: Step-by-step agent-like reasoning framework

### Models Evaluated

- **llama-3.1-8b-instant** (Fast, resource-constrained): 200 questions × 8 strategies = 1,600 generations
- **llama-3.3-70b-versatile** (Larger, higher capability): 200 questions × 8 strategies = 1,600 generations

## Running Experiments

### Quick Baseline Test (50 questions)
```bash
python fast_test_experiment.py
# Output: fast_test_results.csv, evaluated_results.csv
```

### Full Evaluation (200 questions, 8B model)
```bash
python medium_test_experiment.py
# Output: medium_test_results_200_8b.csv
python evaluate_results.py
# Output: evaluated_results_200_8b.csv
```

### Full Evaluation (200 questions, 70B model)
```bash
python medium_test_experiment.py --model llama-3.3-70b-versatile
# Output: medium_test_results_200_70b.csv
python evaluate_results.py --results medium_test_results_200_70b.csv
# Output: evaluated_results_200_70b.csv
```

## Analysis & Results

### Generate Human Evaluation Set
```bash
python generate_human_responses.py
# Output: human_average_scores.csv, human_evaluation_real_5_annotators.csv
```

### Run Statistical Tests
```bash
python run_stats_tests.py
# Outputs: stats_anova_quality_score.txt, stats_tukey_quality_score.csv
```

### Comparative Analysis
```bash
python analyze_human_vs_llm.py
# Output: human_vs_llm_full_comparison.csv
```

### Generate All Visualizations & Tables
```bash
python generate_all_paper_figures.py
# Generates all publication-ready figures and tables
```

## Output Files

### Result Files
- `evaluated_results_200_8b.csv` - 8B model results with quality scores
- `evaluated_results_200_70b.csv` - 70B model results with quality scores
- `human_vs_llm_full_comparison.csv` - Human performance vs all LLM strategies

### Analysis Tables
- `main_results_table.csv` - Summary of all techniques across models
- `table_cost_efficiency.csv` - Token usage and latency analysis
- `table_variance.csv` - Variance metrics per strategy
- `table_error_analysis.csv` - Error categorization by technique
- `table_statistical_significance.csv` - ANOVA and Tukey results
- `table_8b_vs_70b_comparison.csv` - Model size comparison

### LaTeX Tables (for publication)
- `main_results_table.tex`
- `table_cost_efficiency.tex`
- `table_variance.tex`
- `table_error_analysis.tex`
- `table_statistical_significance.tex`

## Key Findings

- Statistical significance testing via ANOVA and Tukey HSD
- Cost-quality tradeoff analysis for resource-constrained deployment
- Model-specific strategy effectiveness (8B vs 70B models)
- Human-LLM quality comparison across all techniques
- Error pattern analysis by prompting strategy

## Dependencies

```
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scipy>=1.10.0          # Statistical tests
statsmodels>=0.14.0    # Advanced statistics
groq>=0.9.0            # Groq API client
streamlit>=1.36.0      # Dashboard/visualization
```

## Key Configuration Files

- **prompt_templates.py** - Define and modify prompting strategies here
- **requirements.txt** - Add/update dependencies
- **.gitignore** - Exclude sensitive data, API keys, and large datasets

## Usage Notes

1. **API Keys**: Store in environment variables, never commit to version control
2. **Large Files**: CSV results (8-16MB) may require git-lfs for full repository
3. **Computation Time**: Full evaluation (3,200 LLM calls) takes 4-6 hours
4. **Parallelization**: Experiments can be parallelized via multiple terminals with different models/subsets

## License

[Add your license information here]

## Contact & Attribution

[Add contact information if applicable]

## Project Timeline

- **Dataset Creation**: Complete
- **Baseline Experiments** (50 samples): Complete
- **Full Scale Experiments** (200 samples, 8B): Complete
- **Full Scale Experiments** (200 samples, 70B): Complete
- **Human Evaluation**: Complete
- **Statistical Analysis**: Complete
- **Publication Preparation**: In Progress

---

**Last Updated**: May 2026
