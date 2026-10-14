# Bilingual Mental Health Dataset (BMHD)

[![Paper](https://img.shields.io/badge/Paper-Annals%20of%20Data%20Science-blue)](https://link.springer.com/article/10.1007/s40745-026-00743-y)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs40745--026--00743--y-informational)](https://doi.org/10.1007/s40745-026-00743-y)

## Overview

The **Bilingual Mental Health Dataset (BMHD)** is a curated dataset designed to support research on mental health text classification, cross-lingual modelling, and interpretability analysis.

The dataset contains aligned **English and Bengali** textual data collected from Reddit, covering ten mental health conditions and a neutral category. It is intended for use in natural language processing (NLP) tasks, particularly in low-resource and multilingual settings.

BMHD was introduced in:

> Faisal, M.R., Meem, S.I. & Uddin, S. Understanding Cross‑Lingual Model Behavior in Mental Health NLP: Evidence from English–Bengali Transformer Models. *Annals of Data Science* (2026). https://doi.org/10.1007/s40745-026-00743-y  
> 📄 [Read the paper on Springer Link](https://link.springer.com/article/10.1007/s40745-026-00743-y) (Open Access)

---

## Authors

- **Moshiur Rahman Faisal**, North South University, Dhaka, Bangladesh  
- **Suraiya Islam Meem**, University of Wollongong, Wollongong, Australia  
- **Shahadat Uddin**\*, School of Electrical and Computer Engineering, Faculty of Engineering, University of Sydney, NSW, Australia  

\* Corresponding author

---

## Dataset Composition

- **Total Samples (English):** 10,672  
- **Total Samples (Bengali):** 10,672  
- **Languages:** English, Bengali  
- **Structure:** Parallel (aligned at sample level)  

Each Bengali instance is a translated and validated version of its English counterpart.

---

## Classes

The dataset includes **11 categories** (10 mental health classes + 1 neutral class):

- Addiction  
- Alcoholism  
- Anxiety  
- Asperger's
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

Data was collected via Reddit's official API from publicly available posts across mental health–related subreddits (initial corpus of ~12,000 English posts). These communities provide real-world, user-generated content reflecting personal experiences, symptoms, and coping strategies.

---

## Preprocessing

The dataset underwent several preprocessing steps:

- Manual removal of noise, irrelevant content, and inconsistencies  
- Correction of spelling, incomplete words, and grammatical errors  
- Removal of HTML tags, special characters, and non-textual elements (NLTK)  
- Text normalisation for consistency  

After preprocessing, the English corpus contained 10,672 posts.

---

## Translation and Alignment

To create the Bengali dataset:

- English texts were translated using the Google Translation API  
- Manual review removed out-of-context or semantically inconsistent sentences  
- BNLP toolkit was used to remove non-Bengali characters and noisy tokens  

### Translation Quality

**Automated metrics**, comparing back-translated English against the original:

- **Jaccard Similarity:** 0.9997  
- **BLEU Score:** 0.999  
- **ROUGE-L Score:** 0.9998  

**Independent human evaluation** was additionally conducted on a stratified sample of 1,066 pairs (~10% of the dataset), balanced across class and sequence length (33 strata), and rated by two independent evaluators on a 5-point semantic equivalence scale:

- **Mean Rating:** 4.675 / 5 (SD = 0.592)  
- **94.5%** of pairs rated 4 or 5  

Known translation issues noted by evaluators: some terms (e.g. "boyfriend", "autism") were transliterated rather than translated; occasional missing punctuation or omitted words; informal/colloquial expressions were harder to translate. Quality declined slightly for longer sequences. Raw human evaluation data is included in this repository.

---

## Annotation Process

A two-stage annotation framework was applied:

1. **Initial Labelling**  
   - Based on subreddit categories  

2. **Multi-Annotator Review**  
   - Four NLP-experienced annotators independently labelled the data  
   - Final labels determined using majority voting  

Samples with persistent disagreement or unclear labels were removed to ensure high data quality.

### Inter-Annotator Agreement

- **English:** Cohen's Kappa = 0.91  
- **Bengali:** Cohen's Kappa = 0.89  

---

## Data Characteristics

- The dataset is **largely balanced** across classes; Asperger's and Self Harm have comparatively fewer samples  
- Bengali texts show slightly higher token lengths and vocabulary diversity  
- Neutral posts are the shortest (median 13 tokens); Bipolar posts are the longest (up to 5,600 tokens in English)  
- Mental health categories exhibit varying linguistic patterns, useful for interpretability research  

---

## Reproducibility

- Train/test split: 80/20, stratified by class, fixed seed (42), performed once at the post level so parallel English–Bengali pairs never split across partitions  
- Hyperparameters selected per model and setting via Optuna (10 trials, seed 42)  
- Final models trained across 3 independent seeds (42, 123, 2024), with results reported as mean ± standard deviation  
- Maximum input length 512 subword tokens with head truncation  

---

## Benchmark Results

Best macro-F1 per setting (mean ± SD across 3 seeds):

| Setting   | Best Model   | Macro-F1          |
|-----------|--------------|-------------------|
| English   | BERT Base    | 0.7814 ± 0.0068   |
| Bengali   | BanglaBERT   | 0.7254 ± 0.0055   |
| Bilingual | XLM-RoBERTa  | 0.7493 ± 0.0041   |

Models evaluated: BERT Base, Multilingual BERT, BanglaBERT, Multilingual ModernBERT (mmBERT), XLM-RoBERTa. Full results, zero-shot transfer, and class-wise analysis are in the paper.

---

## Interpretability Analysis

The dataset supports cross-lingual interpretability research using multiple attribution methods:

- **Integrated Gradients** — cross-lingual attribution overlap between parallel English/Bengali predictions  
- **SHAP** and **LIME** — cross-method agreement analysis on token-level explanations  

This enables systematic comparison of *why* models predict as they do, across languages and across explanation methods, not just classification accuracy alone.

**Key findings from the paper:**

- Models perform consistently worse on Bengali despite identical underlying content  
- Reduced sensitivity to high-risk suicidal and psychosis-related cues in Bengali  
- Depression and Suicidal Thought are the most confused classes in every setting  
- Agreement between IG, SHAP, and LIME is low, and lower for Bengali than English  

---

## Limitations

- Bengali text is machine-translated from English, not natively authored  
- Labels were not verified by licensed mental health professionals  
- Bilingual, not code-mixed (Banglish) text  
- Reddit-only source; patterns may not generalise to other platforms  

---

## Ethical Considerations

- All data is sourced from **publicly available content**  
- Personally identifiable information (PII), including usernames and URLs, has been removed  
- Labels reflect self-expressed content, not medically confirmed diagnoses  
- No clinical diagnoses are inferred or validated  

> ⚠️ This dataset is intended **for research purposes only** and should not be used for clinical decision-making.

---

## Potential Use Cases

- Mental health text classification  
- Cross-lingual and multilingual NLP  
- Low-resource language modelling  
- Model interpretability and explainability (attribution-based and cross-method agreement)  
- Transfer learning and zero-shot cross-lingual transfer research

---

## Citation

If you use this dataset, please cite:

> Faisal, M.R., Meem, S.I. & Uddin, S. Understanding Cross‑Lingual Model Behavior in Mental Health NLP: Evidence from English–Bengali Transformer Models. *Annals of Data Science* (2026). https://doi.org/10.1007/s40745-026-00743-y

### BibTeX

```bibtex
@article{Faisal2026,
  author  = {Faisal, Moshiur Rahman and Meem, Suraiya Islam and Uddin, Shahadat},
  title   = {Understanding Cross-Lingual Model Behavior in Mental Health {NLP}: Evidence from {English}--{Bengali} Transformer Models},
  journal = {Annals of Data Science},
  year    = {2026},
  issn    = {2198-5812},
  doi     = {10.1007/s40745-026-00743-y},
  url     = {https://doi.org/10.1007/s40745-026-00743-y}
}
```

---

## Acknowledgements

We thank Samin Zaman, Fahim Istiak, and Ali Newaz Joarder for their contributions to translating English text into Bengali and supporting the quality assurance of the bilingual dataset.
