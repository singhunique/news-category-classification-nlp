**Automated News Category Classification using NLP**

An AI application that automatically classifies news articles into topic categories (Politics, Sports, Business, Entertainment, Science, Travel) from their headline and short description, using TF-IDF + Logistic Regression as the primary model, benchmarked against Sentence-BERT embeddings.

**Business Problem

Digital news publishers and aggregators process a high volume of articles daily. Manual categorisation by editors is slow, inconsistent, and doesn't scale as publication volume grows. Automating category assignment speeds up content organisation, search, and recommendation, while freeing editorial time for higher-value work.

**Dataset
Source: News Category Dataset (Misra, 2022)
Original: 209,527 articles across 42 categories (headline, short description, category, author, date, link)
Scope used: 6 categories — Politics, Sports, Business, Entertainment, Science, Travel
After filtering to these 6 categories: 76,139 articles
After deduplication: 67,964 articles (32,427 Politics, 14,772 Entertainment, 9,418 Travel, 5,130 Business, 4,414 Sports, 1,803 Science) — notably imbalanced, addressed via class_weight="balanced"
Pipeline
Dataset → Data Cleaning → Category Filtering → Text Combination →
Train/Test Split (80/20, stratified) → TF-IDF → Logistic Regression → Prediction → Evaluation
Columns kept: headline, short_description, category
Cleaning: HTML entity decoding, HTML/URL stripping, whitespace normalisation, duplicate removal
Feature combination: cleaned headline + cleaned description merged into combined_text
Feature extraction: TF-IDF, unigrams + bigrams, English stopwords removed, min document frequency of 2 → 50,000 features
Primary model: Logistic Regression (class_weight="balanced") — chosen as an efficient, interpretable baseline for high-dimensional sparse text features
Comparison model: Sentence-BERT (all-MiniLM-L6-v2) embeddings + Logistic Regression, trained on a 30,000-article subset (for computational tractability)
Results
Model	Accuracy	Macro-F1
TF-IDF + Logistic Regression (full 6-category set)	88.44%	83.86%
Sentence-BERT + Logistic Regression (30K subset)	87.52%	82.37%

Per-category F1 (TF-IDF model): Politics 0.9190, Travel 0.8980, Entertainment 0.8913, Sports 0.8498, Science 0.7435, Business 0.7302

Key experiments:

Category count effect (RQ1): accuracy drops as more categories are added — 97.61% (2 categories) → 90.08% (4 categories) → 88.44% (6 categories)
Category confusion (RQ2): most errors occur between Politics↔Business and Politics↔Entertainment, reflecting genuine topical overlap (e.g. government/business policy coverage)
Headline-only vs. headline+description (RQ3): adding the short description improves accuracy from 85.14% to 88.44% and macro-F1 from 79.23% to 83.86%
Repository Structure
├── NLP.ipynb          # Primary file — full pipeline, all experiments, all outputs
├── NLP.py             # Script export of the same notebook (for running outside Jupyter)
├── NLP.html           # Static rendered export of the notebook (viewable in a browser, no setup needed)
├── requirements.txt   # Python dependencies
└── README.md

NLP.ipynb is the source of truth. NLP.py and NLP.html are convenience exports of the exact same code/results — a plain-script version and a quick-view version — not separate implementations.

How to Run
Option A: Google Colab (recommended, no local setup)
Go to colab.research.google.com → File → Open notebook → GitHub tab → paste this repo's URL, or upload NLP.ipynb directly.
Download the dataset from Kaggle and upload the JSON file to the Colab file browser (left sidebar).
Add a cell at the top: !pip install sentence-transformers and run it.
Runtime → Run all.
Option B: Local Jupyter
Clone the repo:
bash
   git clone https://github.com/singhunique/news-category-classification-nlp.git
   cd news-category-classification-nlp
Install dependencies:
bash
   pip install -r requirements.txt
Download the dataset from Kaggle and place the JSON file in the project's root folder.
Launch Jupyter and run:
bash
   jupyter notebook

Open NLP.ipynb, then Kernel → Restart & Run All.

Limitations
Sentence-BERT was trained on a 30K-article subset (vs. the full ~68K for TF-IDF), so the model comparison is not fully controlled
Class imbalance remains a factor even with balanced class weights (Science: 1,803 articles vs. Politics: 32,427)
Scope is limited to 6 categories from a single publisher (HuffPost); generalisation to other publishers/topics is untested
Business Recommendation

Deploy as an editorial support tool rather than a fully automated system: high-confidence predictions can be auto-tagged, while low-confidence predictions are routed to human editors for review. This balances the ~88% accuracy against the cost of miscategorisation on ambiguous articles.

References

Full citation list is in the accompanying report (Harvard style). Key sources include Misra (2022) for the dataset, Salih et al. (2025) on BERT fine-tuning for news classification, and Fields et al. (2024) on transformer-based text classification.

Author

Harsimran Singh — individual project for B198c7 AI Applications for Digital Business, Gisma University of Applied Sciences.
