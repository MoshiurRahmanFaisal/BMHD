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
- Asperger’s Syndrome  
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

Semantic alignment was validated using:

- **Jaccard Similarity:** 0.9997  
- **BLEU Score:** 0.999  
- **ROUGE-L Score:** 0.9998  

These results indicate near-perfect preservation of meaning.

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

- **English:** Cohen’s Kappa = 0.91  
- **Bengali:** Cohen’s Kappa = 0.89  

---

## Data Characteristics

- The dataset is **largely balanced** across classes  
- Bengali texts show slightly higher token lengths and vocabulary diversity  
- Mental health categories exhibit varying linguistic patterns, useful for interpretability research  

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
- Model interpretability and explainability  
- Transfer learning research  
