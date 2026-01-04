# Topic Modelling Efficacy: A Comparative Study

A comprehensive comparative analysis of topic modeling techniques on real-world consumer complaint data. This project evaluates three distinct approaches—**LDA (Latent Dirichlet Allocation)**, **BERTopic**, and **LLM-based topic extraction** (using Deepseek and Llama)—to understand their strengths, limitations, and practical applicability.

## Overview

Topic modeling is a fundamental technique for discovering abstract themes and patterns in large text collections. This study compares classical statistical models with modern neural and LLM-based approaches across multiple coherence and diversity metrics to determine which methods best capture meaningful topics in consumer complaint narratives.

### Dataset
- **Source:** [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data)
- **Content:** Consumer complaint narratives from financial institutions
- **Size:** ~19,656 documents evaluated across models
- **Focus Domain:** Financial services complaints (credit reporting, debt collection, loan issues, etc.)

## Methodology

### 1. **LDA (Latent Dirichlet Allocation)**
Traditional probabilistic graphical model for topic discovery.
- Implemented using Gensim
- Generates interpretable word distributions for each topic
- Suitable for understanding statistical topic-document relationships

### 2. **BERTopic**
Modern neural embedding-based approach leveraging transformer models.
- Uses BERT embeddings for semantic understanding
- Clustering-based topic detection
- Captures semantic relationships beyond word co-occurrence

### 3. **LLM-Based Extraction**
Generative approach using large language models (Deepseek and Llama).
- **Approach:** Per-row topic extraction instead of global topic lists
- **Rationale:** Overcomes context length constraints of LLMs by processing each complaint independently
- **Output:** 5 distinct topics per complaint narrative
- **Advantages:** 
  - Handles large datasets without degraded quality
  - Generates diverse, interpretable topics
  - No inherent context window limitations across the full corpus

#### Why Per-Row Processing?
Initial experiments with extracting global topic lists failed due to context limitations. When processing > 100 rows simultaneously, LLMs produced vague or incoherent topics. The per-row approach ensures:
- Consistent output quality across all documents
- Richer topic diversity
- Scalability to larger datasets
- Per-document topic customization

## Results

### Coherence Metrics
| Model | C_V | U_MASS | C_NPMI | C_UCI |
|-------|-----|--------|--------|-------|
| **LDA** | 0.5750 | -1.2596 | 0.0763 | 0.3734 |
| **BERTopic** | 0.5643 | -0.4040 | 0.0699 | 0.2519 |

**Interpretation:**
- **C_V (Coherence_V):** Measures topic coherence using normalized PMI. LDA slightly outperforms BERTopic (0.575 vs 0.564)
- **U_MASS:** Intrinsic coherence measure. Lower (more negative) indicates better topic separation. LDA shows better separation.
- **C_NPMI & C_UCI:** Additional coherence measures confirming LDA's marginal advantage

### Diversity Metrics
| Model | Topic Diversity | Avg Pairwise Jaccard Distance |
|-------|-----------------|-------------------------------|
| **LDA** | 0.3500 | 0.9110 |
| **BERTopic** | 0.4538 | 0.9016 |

**Interpretation:**
- BERTopic generates more diverse topics (0.454 vs 0.350)
- Both models maintain high term uniqueness across topics (Jaccard > 0.90)
- Trade-off: coherence vs diversity

## Project Structure

```
Topic-Modelling-Efficacy/
├── BERTopic_code.ipynb           # BERTopic implementation
├── LDA_code.ipynb                 # LDA implementation with Gensim
├── llm_code.py                    # Deepseek API-based topic extraction
├── LLM_metrics_code.py            # Metric computation for LLM results
├── Bertopic_metrics.txt           # BERTopic evaluation results
├── LDA_metrics.txt                # LDA evaluation results
├── Deepseek_metrics.txt           # Deepseek model metrics
├── llama_metrics.json             # Llama model metrics
├── LLM Sample Output.txt          # Sample topics from Deepseek
├── Reason_for_LLM_output_format.txt # Technical rationale for LLM approach
├── dataset link.txt               # Data source reference
└── README.md                      # This file
```

## Key Findings

### Classical vs Modern Approaches
1. **LDA** achieves slightly higher coherence, indicating mathematically consistent topic-word distributions
2. **BERTopic** provides greater topic diversity through semantic clustering
3. **LLM-based extraction** offers interpretable, domain-specific topics with full dataset scalability

### Trade-offs
- **Coherence Focus:** LDA optimal for statistically coherent topics
- **Diversity Focus:** BERTopic for varied, distinct topic vocabularies  
- **Interpretability & Scale:** LLMs for human-readable, domain-adapted topics across large corpora

## Technical Implementation

### Dependencies
- **LDA/BERTopic:** Gensim, scikit-learn, transformers, BERTopic library
- **LLM Extraction:** OpenRouter API (Deepseek R1), requests library, pandas
- **Metrics:** Gensim coherencemodel, custom diversity computations

### Evaluation Metrics
All models evaluated on:
- **Coherence Scores:** C_V, U_MASS, C_NPMI, C_UCI
- **Diversity Scores:** Topic diversity, pairwise Jaccard distance
- **Corpus-level:** Dictionary size, document count, topic coverage

## Usage

### Running LDA
Open `LDA_code.ipynb` and execute cells sequentially to train the LDA model and generate metrics.

### Running BERTopic
Open `BERTopic_code.ipynb` and execute for semantic clustering-based topic discovery.

### Running LLM-based Extraction
1. Set OpenRouter API key in `env.local`:
   ```
   OPEN_ROUTER_KEY="your_api_key_here"
   ```
2. Configure parameters in `llm_code.py`:
   - `START_ROW` / `END_ROW`: Row range to process
   - `PROCESS_CHUNK_SIZE`: Batch size for API calls
   - `MODEL`: Deepseek or compatible LLM model ID
3. Run: `python llm_code.py`
4. Compute metrics: `python LLM_metrics_code.py --rows_file combined_rows.txt --csv_file filtered_complaints_narratives.csv`

## Conclusion

This comparative study demonstrates that **no single approach dominates**. The choice of topic modeling technique depends on use-case priorities:
- **Statistical rigor:** Use LDA
- **Semantic richness:** Use BERTopic  
- **Interpretability & flexibility:** Use LLM-based extraction

For production systems, a **hybrid approach** combining strengths of multiple models may provide optimal results.
