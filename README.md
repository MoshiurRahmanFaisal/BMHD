# Bilingual Mental Health Dataset (BMHD)

## Overview

The **Bilingual Mental Health Dataset (BMHD)** is a curated dataset designed to support research on mental health text classification, cross-lingual modelling, and interpretability analysis.

The dataset contains aligned **English and Bengali** textual data collected from Reddit, covering a range of mental health conditions and a neutral category. It is intended for use in natural language processing (NLP) tasks, particularly in low-resource and multilingual settings.

---

## Dataset Composition

- **Total Samples (English):** 10,672  
- **Total Samples (Bengali):** 10,672  
- **Languages:** English, Bengali  
- **Structure:** Parallel (aligned at sample level)  

Each Bengali instance is a translated and validated version of its English counterpart.

---

## Classes

The dataset includes **11 categories**:

- Addiction  
- Alcoholism  
- Anxiety  
- Asperger's Syndrome  
- Bipolar Disorder  
- Borderline Personality Disorder  
- Depression  
- Schizophrenia  
- Self Harm  
- Suicidal Thought  
- Neutral  

The **Neutral** class contains non-mental-health-related content and serves as a baseline for classification.

---

## Data Source

Data was collected from publicly available Reddit posts across mental health–related subreddits. These communities provide real-world, user-generated content reflecting personal experiences, symptoms, and coping strategies.

---

## Preprocessing

The dataset underwent several preprocessing steps:

- Removal of noise, irrelevant content, and inconsistencies  
- Correction of spelling and grammatical errors  
- Removal of HTML tags and special characters  
- Text normalisation for consistency  

After preprocessing, the dataset was standardised for downstream NLP tasks.

---

## Translation and Alignment

To create the Bengali dataset:

- English texts were translated using the Google Translation API  
- Manual review ensured contextual and semantic consistency  
- BNLP toolkit was used for additional cleaning  

### Translation Quality

**Automated metrics**, comparing back-translated English against the original:

- **Jaccard Similarity:** 0.9997  
- **BLEU Score:** 0.999  
- **ROUGE-L Score:** 0.9998  

**Independent human evaluation** was additionally conducted on a stratified sample of 1,066 pairs (~10% of the dataset), balanced across class and sequence length, and rated by two independent evaluators on a 5-point semantic equivalence scale:

- **Mean Rating:** 4.675 / 5 (SD = 0.592)  
- **94.5%** of pairs rated 4 or 5  

Together, these results indicate near-perfect preservation of meaning, confirmed both by automated metrics and independent human judgement.

---

## Annotation Process

A two-stage annotation framework was applied:

1. **Initial Labelling**  
   - Based on subreddit categories  

2. **Expert Annotation**  
   - Four annotators independently labelled the data  
   - Final labels determined using majority voting  

Ambiguous samples were removed to ensure high data quality.

### Inter-Annotator Agreement

- **English:** Cohen's Kappa = 0.91  
- **Bengali:** Cohen's Kappa = 0.89  

---

## Data Characteristics

- The dataset is **largely balanced** across classes  
- Bengali texts show slightly higher token lengths and vocabulary diversity  
- Mental health categories exhibit varying linguistic patterns, useful for interpretability research  

---

## Reproducibility

- Train/test split: 80/20, stratified by class, fixed seed, performed once at the post level so parallel English–Bengali pairs never split across partitions  
- Final models trained across 3 independent seeds, with results reported as mean ± standard deviation  

---

## Interpretability Analysis

The dataset supports cross-lingual interpretability research using multiple attribution methods:

- **Integrated Gradients** — cross-lingual attribution overlap between parallel English/Bengali predictions  
- **SHAP** and **LIME** — cross-method agreement analysis on token-level explanations  

This enables systematic comparison of *why* models predict as they do, across languages and across explanation methods, not just classification accuracy alone.

---

## Ethical Considerations

- All data is sourced from **publicly available content**  
- Personally identifiable information (PII) has been removed  
- No clinical diagnoses are inferred or validated  

> ⚠️ This dataset is intended **for research purposes only** and should not be used for clinical decision-making.

---

## Potential Use Cases

- Mental health text classification  
- Cross-lingual and multilingual NLP  
- Low-resource language modelling  
- Model interpretability and explainability (attribution-based and cross-method agreement)  
- Transfer learning and zero-shot cross-lingual transfer research
