#!/usr/bin/env python
# coding: utf-8

# # Install Required Libraries

# In[1]:


#!pip install pandas numpy matplotlib scikit-learn beautifulsoup4 sentence-transformers -q


# ## Import Libraries

# In[2]:


import pandas as pd
import numpy as np

import re
import html

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_colwidth", 150)

RANDOM_STATE = 42


# # Dataset Path

# In[3]:


DATA_PATH = "News_Category_Dataset_v3.json"

print("Dataset path:", DATA_PATH)


# # Load JSON Lines Dataset

# In[4]:


df = pd.read_json(DATA_PATH, lines=True)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# # Inspect Dataset

# In[5]:


print("First 5 records:")
display(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()


# # Check Missing Values and Duplicates

# In[6]:


print("Missing values:")
display(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique categories:")
print(df["category"].nunique())


# # Explore All Categories

# In[7]:


category_counts = df["category"].value_counts()

print("Total number of categories:", len(category_counts))

display(category_counts)


# In[8]:


plt.figure(figsize=(12, 8))

category_counts.sort_values().plot(kind="barh")

plt.title("Distribution of All News Categories")
plt.xlabel("Number of Articles")
plt.ylabel("Category")

plt.tight_layout()
plt.show()


# # Keep Relevant Columns

# In[9]:


df = df[
    [
        "headline",
        "short_description",
        "category"
    ]
].copy()

display(df.head())


# # Filter the Six Selected Categories

# In[10]:


selected_categories = [
    "POLITICS",
    "SPORTS",
    "BUSINESS",
    "ENTERTAINMENT",
    "SCIENCE",
    "TRAVEL"
]

df_filtered = df[
    df["category"].isin(selected_categories)
].copy()

print("Original dataset shape:", df.shape)
print("Filtered dataset shape:", df_filtered.shape)

display(
    df_filtered["category"]
    .value_counts()
)


# # Clean Missing and Empty Values

# In[11]:


df_filtered["headline"] = (
    df_filtered["headline"]
    .fillna("")
    .astype(str)
)

df_filtered["short_description"] = (
    df_filtered["short_description"]
    .fillna("")
    .astype(str)
)

df_filtered["category"] = (
    df_filtered["category"]
    .fillna("")
    .astype(str)
)

df_filtered = df_filtered[
    (df_filtered["headline"].str.strip() != "") &
    (df_filtered["short_description"].str.strip() != "") &
    (df_filtered["category"].str.strip() != "")
].copy()

print("Dataset after removing missing/empty values:")
print(df_filtered.shape)


# # Text Cleaning Function

# In[12]:


def clean_text(text):

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# In[13]:


df_filtered["clean_headline"] = (
    df_filtered["headline"]
    .apply(clean_text)
)

df_filtered["clean_description"] = (
    df_filtered["short_description"]
    .apply(clean_text)
)


# # Combine Headline and Description

# In[14]:


df_filtered["combined_text"] = (
    df_filtered["clean_headline"]
    + ". "
    + df_filtered["clean_description"]
)

display(
    df_filtered[
        [
            "clean_headline",
            "clean_description",
            "combined_text",
            "category"
        ]
    ].head()
)


# # Remove Duplicate Articles

# In[15]:


print(
    "Duplicates before removal:",
    df_filtered.duplicated(
        subset=["combined_text", "category"]
    ).sum()
)

df_filtered = df_filtered.drop_duplicates(
    subset=["combined_text", "category"]
).reset_index(drop=True)

print("Final cleaned dataset shape:", df_filtered.shape)


# # Visualise Final Six-Category Distribution

# In[16]:


final_counts = (
    df_filtered["category"]
    .value_counts()
    .sort_values(ascending=False)
)

display(final_counts)

plt.figure(figsize=(9, 5))

final_counts.plot(kind="bar")

plt.title("Distribution of Selected News Categories")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# # Export Cleaned Data to CSV

# In[17]:


output_columns = [
    "clean_headline",
    "clean_description",
    "combined_text",
    "category"
]

df_filtered[output_columns].to_csv(
    "cleaned_news_category_dataset.csv",
    index=False
)

print("Cleaned dataset exported successfully.")


# # Train/Test Split for Main Model

# In[18]:


X = df_filtered["combined_text"]
y = df_filtered["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining percentage:")
print(round(len(X_train) / len(X) * 100, 2), "%")

print("\nTesting percentage:")
print(round(len(X_test) / len(X) * 100, 2), "%")


# # TF-IDF Feature Extraction

# In[19]:


tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# # Train Logistic Regression

# In[20]:


logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

print("Logistic Regression model trained successfully.")


# # Evaluate Main Model

# In[21]:


y_pred = logistic_model.predict(
    X_test_tfidf
)

main_accuracy = accuracy_score(
    y_test,
    y_pred
)

main_macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("MAIN MODEL RESULTS")
print("------------------")

print(
    f"Accuracy: {main_accuracy:.4f}"
)

print(
    f"Macro F1 Score: {main_macro_f1:.4f}"
)


# In[22]:


print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# # Confusion Matrix: RQ2

# In[23]:


labels = selected_categories

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "Confusion Matrix - TF-IDF + Logistic Regression"
)

plt.tight_layout()
plt.show()


# In[24]:


confusion_records = []

for i, actual in enumerate(labels):

    for j, predicted in enumerate(labels):

        if i != j:

            confusion_records.append(
                {
                    "Actual": actual,
                    "Predicted": predicted,
                    "Count": cm[i, j]
                }
            )

confusion_df = pd.DataFrame(
    confusion_records
).sort_values(
    "Count",
    ascending=False
)

display(confusion_df.head(10))


# # RQ1: Effect of Number of Categories

# In[25]:


category_experiments = {

    "2 Categories": [
        "POLITICS",
        "SPORTS"
    ],

    "4 Categories": [
        "POLITICS",
        "SPORTS",
        "BUSINESS",
        "ENTERTAINMENT"
    ],

    "6 Categories": [
        "POLITICS",
        "SPORTS",
        "BUSINESS",
        "ENTERTAINMENT",
        "SCIENCE",
        "TRAVEL"
    ]
}


# In[26]:


rq1_results = []

for experiment_name, categories in category_experiments.items():

    experiment_df = df_filtered[
        df_filtered["category"].isin(categories)
    ]

    X_exp = experiment_df["combined_text"]
    y_exp = experiment_df["category"]

    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
        X_exp,
        y_exp,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_exp
    )

    vectorizer_exp = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer_exp.fit_transform(
        X_train_exp
    )

    X_test_vec = vectorizer_exp.transform(
        X_test_exp
    )

    model_exp = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model_exp.fit(
        X_train_vec,
        y_train_exp
    )

    pred_exp = model_exp.predict(
        X_test_vec
    )

    accuracy_exp = accuracy_score(
        y_test_exp,
        pred_exp
    )

    macro_f1_exp = f1_score(
        y_test_exp,
        pred_exp,
        average="macro"
    )

    rq1_results.append(
        {
            "Experiment": experiment_name,
            "Number_of_Categories": len(categories),
            "Number_of_Articles": len(experiment_df),
            "Accuracy": accuracy_exp,
            "Macro_F1": macro_f1_exp
        }
    )

rq1_results_df = pd.DataFrame(
    rq1_results
)

display(rq1_results_df)


# In[27]:


plt.figure(figsize=(7, 5))

plt.plot(
    rq1_results_df["Number_of_Categories"],
    rq1_results_df["Accuracy"],
    marker="o"
)

plt.title(
    "Classification Accuracy vs Number of Categories"
)

plt.xlabel(
    "Number of Categories"
)

plt.ylabel(
    "Accuracy"
)

plt.xticks(
    rq1_results_df["Number_of_Categories"]
)

plt.grid(True)

plt.tight_layout()
plt.show()


# # RQ3: Headline vs Headline + Description

# In[28]:


def evaluate_text_source(text_column, experiment_name):

    X = df_filtered[text_column]
    y = df_filtered["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(
        X_train
    )

    X_test_vec = vectorizer.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model.fit(
        X_train_vec,
        y_train
    )

    predictions = model.predict(
        X_test_vec
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    return {
        "Input": experiment_name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1
    }


# In[29]:


headline_result = evaluate_text_source(
    "clean_headline",
    "Headline Only"
)

combined_result = evaluate_text_source(
    "combined_text",
    "Headline + Description"
)

rq3_results = pd.DataFrame(
    [
        headline_result,
        combined_result
    ]
)

display(rq3_results)


# In[30]:


rq3_results.set_index(
    "Input"
)[
    ["Accuracy", "Macro_F1"]
].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Headline Only vs Headline + Description"
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model Input"
)

plt.xticks(
    rotation=0
)

plt.ylim(
    0,
    1
)

plt.tight_layout()
plt.show()


# # Optional Sentence-BERT Model

# In[31]:


from sentence_transformers import SentenceTransformer


# In[32]:


sbert_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# In[33]:


SBERT_SAMPLE_SIZE = min(
    30000,
    len(df_filtered)
)

sbert_df = df_filtered.sample(
    n=SBERT_SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

X_sbert = sbert_df["combined_text"]
y_sbert = sbert_df["category"]

X_train_sbert, X_test_sbert, y_train_sbert, y_test_sbert = train_test_split(
    X_sbert,
    y_sbert,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_sbert
)

print(
    "Generating Sentence-BERT embeddings..."
)

X_train_embeddings = sbert_model.encode(
    X_train_sbert.tolist(),
    show_progress_bar=True,
    batch_size=64
)

X_test_embeddings = sbert_model.encode(
    X_test_sbert.tolist(),
    show_progress_bar=True,
    batch_size=64
)


# In[34]:


sbert_classifier = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

sbert_classifier.fit(
    X_train_embeddings,
    y_train_sbert
)

sbert_predictions = sbert_classifier.predict(
    X_test_embeddings
)

sbert_accuracy = accuracy_score(
    y_test_sbert,
    sbert_predictions
)

sbert_macro_f1 = f1_score(
    y_test_sbert,
    sbert_predictions,
    average="macro"
)

print(
    "Sentence-BERT Accuracy:",
    round(sbert_accuracy, 4)
)

print(
    "Sentence-BERT Macro F1:",
    round(sbert_macro_f1, 4)
)


# ## Final Results Table

# In[35]:


final_results = pd.DataFrame(
    {
        "Model": [
            "TF-IDF + Logistic Regression",
            "Sentence-BERT + Logistic Regression"
        ],

        "Accuracy": [
            main_accuracy,
            sbert_accuracy
        ],

        "Macro_F1": [
            main_macro_f1,
            sbert_macro_f1
        ]
    }
)

display(final_results)


# In[36]:


final_results.set_index(
    "Model"
)[
    ["Accuracy", "Macro_F1"]
].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title(
    "Model Performance Comparison"
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.xticks(
    rotation=15
)

plt.ylim(
    0,
    1
)

plt.tight_layout()
plt.show()


# In[ ]:




