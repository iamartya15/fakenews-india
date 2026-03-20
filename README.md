# Fake News Detection in Regional Indian Media

Multilingual NLP system for detecting fake news across Hindi, Odia, Bengali, Tamil, and Telugu.

**Paper:** Submitted to IEEE Access / ACM TALLIP (v2.0, March 2025)
**Author:** Amartya Kushwaha, R.R. Institute of Modern Technology, Lucknow

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 91.3% |
| F1-Score | 91.3% |
| AUC-ROC | 0.961 |

## Languages Covered
Hindi, Bengali, Tamil, Telugu, Odia

## Dataset
12,400 items from AltNews, FactChecker.in, Boom Live, Vishvas News

## Model Architecture
- mBERT fine-tuned (bert-base-multilingual-cased)
- Character-level TF-IDF (n-gram 1-3, vocab 50k)
- Stacked Ensemble: XGBoost + Random Forest + Logistic Regression meta-learner

## Setup
pip install -r requirements.txt

## Citation
Kushwaha, A. (2025). Fake News Detection in Regional Indian Media: A Multilingual NLP and Ensemble Learning Approach. Preprint v2.0.
