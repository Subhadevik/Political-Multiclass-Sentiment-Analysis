# Political-Multiclass-Sentiment-Analysis

A multiclass text classification project that analyzes Tamil-language political social media posts and classifies them into one of seven sentiment/intent categories: **Opinionated**, **Sarcastic**, **Neutral**, **Positive**, **Substantiated**, **Negative**, and **None of the above**.

## Overview

This project preprocesses raw Tamil text scraped from political social media posts, applies Tamil-specific NLP techniques (using the [Stanza](https://stanfordnlp.github.io/stanza/) library), balances the class distribution through upsampling, and trains multiple machine learning classifiers to predict the sentiment/intent label of each post.

## Dataset

- **Training data**: `PS_train.csv` — contains `content` (the Tamil text) and `labels` (one of 7 classes)
- **Test data**: `PS_test_without_lables.csv` — contains `Id` and `content`, used for generating predictions

Initial class distribution (before balancing):

| Label | Count |
|---|---|
| Opinionated | 1361 |
| Sarcastic | 790 |
| Neutral | 637 |
| Positive | 575 |
| Substantiated | 412 |
| Negative | 406 |
| None of the above | 171 |

After upsampling, all 7 classes are balanced to 1361 samples each (9,527 total rows).

## Preprocessing Pipeline

1. **Unicode normalization** — removes zero-width non-joiners (ZWNJ)
2. **Character filtering** — retains only Tamil script characters and spaces, strips numbers and special characters
3. **Whitespace cleanup** — collapses multiple spaces
4. **Spoken-variant normalization** — maps commonly confused vowels, diphthongs, and compound consonants to a standard form (e.g., colloquial vowel substitutions)
5. **Tokenization** — performed using Stanza's Tamil (`ta`) tokenizer pipeline
6. **Class balancing** — upsampling with replacement to match the largest class size

## Models Trained

All models use **TF-IDF vectorization** (max 5000 features) on the preprocessed text. The following classifiers were trained and evaluated:

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Random Forest | 94.9% | 84.0% |
| Decision Tree | 94.9% | 82.9% |
| CatBoost | 70.6% | 60.9% |
| Gradient Boosting | 68.8% | 57.9% |
| Logistic Regression | 62.5% | 51.8% |
| Naive Bayes | 57.5% | 49.5% |
| AdaBoost | 28.9% | 28.7% |

**Random Forest** and **Decision Tree** performed best on the test set, while **AdaBoost** performed poorly relative to the others on this dataset.

## Project Structure
