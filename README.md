# Sentiment Analysis of Amazon Product Reviews Using Machine Learning at Scale

## Overview

Star ratings are a convenient summary of customer feedback but they only tell part of the story. The real substance often lies in the text: what people write when they explain why they liked or disliked something.In this project, we wanted to explore whether a machine learning model could learn to identify **positive vs. negative sentiment based on how people write**, not just what rating they gave. We weren’t trying to predict star ratings directly. Instead, we used them to label our training data (1–3 stars as negative, 4–5 as positive) and focused on **teaching a model to recognize tone, expression, and emotion** in written reviews.

This matters because, at scale, platforms can't read every review and star ratings alone don’t always capture nuance. By modeling how sentiment is expressed in writing, we create a foundation for smarter tools: better feedback analysis, trend detection, review summarization, and more.

Using PySpark on Google Cloud, we processed over 18 million reviews across six Amazon product categories. We engineered custom features from review text, metadata, and sentiment lexicons, and trained multiple classifiers to see how well sentiment could be predicted. Our final model achieved an F1 score of ~87%, showing strong performance and surfacing patterns that go beyond the star.

## Problem Statement

Customer reviews are rich in language, emotion, and personal experience but this is hard to quantify at scale. Most platforms rely on star ratings as a proxy for satisfaction, but those ratings don’t always reflect how someone *feels*. Some 4-star reviews are cautious or mixed. Some 2-star reviews are calmly written. Others don’t match the text at all.

We set out to build a model that could learn from how people **write** when they’re satisfied or disappointed — and use that to classify sentiment as positive or negative. While we used star ratings to supervise the model, the goal was to go beyond ratings and capture the **way** people express sentiment.

We trained and evaluated multiple models using features such as:

- TF-IDF vectors of review text
- Verified purchase, Vine program, product category
- Voting behavior (helpful votes, total votes)
- Temporal features (year, month, day of week)
- Custom sentiment lexicon scores and ratios
- Word2Vec embeddings trained on review text

The result is a scalable sentiment analysis model that reflects how customers communicate opinions — and provides a path toward deeper analysis of feedback.

## Data Sources and Description

- **Dataset**: [Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset)
- **Size**: ~18 million reviews across 6 categories:
  - Apparel
  - Beauty
  - Books
  - Electronics
  - Furniture
  - Mobile Electronics
- **Format**: TSV (tab-separated), per category
- **Fields Used**:
  - review_body, star_rating, review_date
  - product_category, verified_purchase, helpful_votes, total_votes
  - vine (Amazon Vine program), review_headline

No significant missing values were found in core fields. Extremely short reviews (<5 characters) were filtered.

## Approach and Methodology

### 1. Data Ingestion and Cleaning

- Downloaded via Kaggle API, extracted selected categories
- Uploaded TSVs to Google Cloud Storage (GCS)
- Used PySpark for custom parsing (handling embedded newlines, tabs)
- Validated categories, added:
  - `helpful_ratio = helpful_votes / total_votes`
  - `review_length`, `has_votes` (binary), cleaned nulls
- Saved cleaned data in Parquet format for efficiency

### 2. Feature Engineering

**Initial Features:**

| Feature | Description |
|--------|-------------|
| TF-IDF | Sparse representation of review text |
| helpful_ratio | Ratio of helpful to total votes |
| has_votes | Binary: whether review received any votes |
| vine_binary | Binary flag for Amazon Vine program |
| verified_purchase_binary | Binary flag for verified purchase |
| product_category_vec | One-hot encoded product category |
| review_year, review_month, review_dayofweek | Temporal features |

**Added Features:**

| Feature | Description |
|---------|-------------|
| sentiment_score | Count(pos_words) - Count(neg_words) |
| positive_ratio | Proportion of positive words in review |
| negative_ratio | Proportion of negative words in review |
| w2v_vector | Word2Vec embeddings (50-dim) from sampled data |

### 3. Modeling and Experiments

**Goal**: Binary classification  
- Label = 1 (positive): star_rating in [4, 5]  
- Label = 0 (negative): star_rating in [1, 2, 3]

**Models Trained:**
- Logistic Regression (with class weights)
- Random Forest (with downsampling of majority class)

**Hyperparameter Tuning:**
- Logistic:
  - `regParam`: 0.01, 0.1, 1.0
  - `elasticNetParam`: 0.0, 0.5, 1.0
- Random Forest:
  - `numTrees`: 30, 50
  - `maxDepth`: 5, 10

**Train/Val/Test Split**:
- Time-based (70/15/15) to avoid data leakage
- Feature transformations applied on training only

## Results and Key Experiments

| Model               | Dataset     | Accuracy | F1 Score |
|---------------------|-------------|----------|----------|
| Logistic Regression | Validation  | 86.84%   | 87.41%   |
| Logistic Regression | Test        | 86.84%   | 87.42%   |
| Random Forest       | Validation  | 76.62%   | 78.31%   |
| Random Forest       | Test        | 76.68%   | 78.37%   |

### Key Insights:

- **Word2Vec** added semantic context that boosted performance significantly.
- **Lexicon-based sentiment features** performed well despite simplicity.
- **Logistic Regression** outperformed Random Forest and generalized better.
- **Handling class imbalance** (class weighting, downsampling) was critical.

## Infrastructure

- **Cloud Platform**: Google Cloud Platform (GCP)
- **Processing Engine**: PySpark on Google Dataproc
- **Cluster Specs**:
  - Master: `n4-standard-4` (4 vCPU, 16 GB RAM)
  - Workers (2): `n4-standard-2` (2 vCPU, 8 GB RAM)
- **Notebook Interface**: JupyterHub
- **Storage**: Google Cloud Storage (GCS)
- **Setup Note**: "Internal IP only" must be unchecked to allow dataset download

## Timeline and Deliverables

| Milestone      | Summary                                                              |
|----------------|----------------------------------------------------------------------|
| Milestone 0    | Project kickoff, environment setup, dataset selection                |
| Milestone 1    | Data parsing, validation, EDA, class imbalance analysis              |
| Milestone 2    | Initial feature engineering, baseline modeling                       |
| Milestone 3    | Sentiment-aware features, Word2Vec embeddings, hyperparameter tuning |
| Final Report   | Consolidated results, full documentation, replication instructions   |


## Resources

- Dataset: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset
- Tools: PySpark, Gensim, NLTK, Scikit-learn, Pandas
- Platform: Google Cloud Platform (Dataproc + GCS)

## How to Contribute

This project is currently closed-source and not accepting contributions.

## Sample Run / Output

_To be added — include sample predictions, classification output, confusion matrix or interpretability plot._

