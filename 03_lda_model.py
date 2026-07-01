"""
STEP 3: LDA (Latent Dirichlet Allocation)
Library: scikit-learn (LatentDirichletAllocation)
Input:   Bag-of-Words (CountVectorizer) on classically-cleaned text
Output:  10 topics with top keywords + document-topic assignment
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_clean.csv")
texts = df["clean_text_classical"].tolist()

N_TOPICS = 10

# ---- Vectorize: Bag of Words (LDA needs raw counts, not TF-IDF) ----
count_vectorizer = CountVectorizer(
    max_df=0.90,      # drop words in >90% of docs (too generic)
    min_df=5,         # drop words in <5 docs (too rare)
    ngram_range=(1, 2)  # unigrams + bigrams (e.g. "charging station")
)
doc_term_matrix = count_vectorizer.fit_transform(texts)
feature_names = count_vectorizer.get_feature_names_out()
print("Vocabulary size:", len(feature_names))
print("Doc-term matrix shape:", doc_term_matrix.shape)

# ---- Fit LDA ----
lda_model = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    max_iter=25,
    learning_method="batch",
    doc_topic_prior=None,   # alpha, auto = 1/n_topics
    topic_word_prior=None   # beta, auto = 1/n_topics
)
lda_topic_matrix = lda_model.fit_transform(doc_term_matrix)  # shape: (docs, topics)

# ---- Extract top words per topic ----
def get_top_words(model, feature_names, n_top_words=10):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        topics[topic_idx] = top_words
    return topics

lda_topics = get_top_words(lda_model, feature_names, n_top_words=10)

print("\n===== LDA TOPICS =====")
for idx, words in lda_topics.items():
    print(f"Topic {idx}: {', '.join(words)}")

# ---- Assign dominant topic to each document ----
df["lda_topic"] = lda_topic_matrix.argmax(axis=1)
df["lda_topic_confidence"] = lda_topic_matrix.max(axis=1)

# ---- Human-readable topic labels (business interpretation of keywords) ----
lda_topic_labels = {}
for idx, words in lda_topics.items():
    lda_topic_labels[idx] = f"Topic {idx}: " + ", ".join(words[:5])

df["lda_topic_label"] = df["lda_topic"].map(lda_topic_labels)

# ---- Topic prevalence (business-critical: what matters most, by volume) ----
topic_counts = df["lda_topic"].value_counts().sort_index()
print("\n===== TOPIC PREVALENCE (document counts) =====")
for idx in range(N_TOPICS):
    cnt = topic_counts.get(idx, 0)
    pct = cnt / len(df) * 100
    print(f"Topic {idx} ({', '.join(lda_topics[idx][:4])}): {cnt} docs ({pct:.1f}%)")

# ---- Save outputs ----
lda_topic_summary = pd.DataFrame([
    {"topic_id": idx, "top_keywords": ", ".join(words), "doc_count": topic_counts.get(idx, 0),
     "pct_of_corpus": round(topic_counts.get(idx, 0)/len(df)*100, 2)}
    for idx, words in lda_topics.items()
]).sort_values("doc_count", ascending=False)

lda_topic_summary.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/lda_topic_summary.csv", index=False)
df[["review_id", "product_type", "brand", "rating", "feedback_text",
    "lda_topic", "lda_topic_label", "lda_topic_confidence"]].to_csv(
    "C:/Users/saksham singhal/Desktop/EV/files/data/lda_document_topics.csv", index=False)

print("\nSaved: lda_topic_summary.csv, lda_document_topics.csv")

# Save model artifacts for cross-model comparison later
df.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_clean.csv", index=False)
