"""
STEP 4: NMF (Non-negative Matrix Factorization)
==================================================
Library: scikit-learn (NMF)
Input:   TF-IDF matrix on classically-cleaned text
         (NMF works better with TF-IDF than raw counts, unlike LDA,
          because TF-IDF downweights globally common words and NMF's
          factorization directly benefits from that contrast)
Output:  10 topics with top keywords + document-topic assignment
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

df = pd.read_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_clean.csv")
texts = df["clean_text_classical"].tolist()

N_TOPICS = 10

# ---- Vectorize: TF-IDF ----
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.90,
    min_df=5,
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
feature_names = tfidf_vectorizer.get_feature_names_out()
print("Vocabulary size:", len(feature_names))
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# ---- Fit NMF ----
nmf_model = NMF(
    n_components=N_TOPICS,
    random_state=42,
    init="nndsvda",
    max_iter=500
)
nmf_topic_matrix = nmf_model.fit_transform(tfidf_matrix)  # (docs, topics)

def get_top_words(model, feature_names, n_top_words=10):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        topics[topic_idx] = top_words
    return topics

nmf_topics = get_top_words(nmf_model, feature_names, n_top_words=10)

print("\n===== NMF TOPICS =====")
for idx, words in nmf_topics.items():
    print(f"Topic {idx}: {', '.join(words)}")

df["nmf_topic"] = nmf_topic_matrix.argmax(axis=1)
df["nmf_topic_confidence"] = nmf_topic_matrix.max(axis=1)

nmf_topic_labels = {idx: f"Topic {idx}: " + ", ".join(words[:5]) for idx, words in nmf_topics.items()}
df["nmf_topic_label"] = df["nmf_topic"].map(nmf_topic_labels)

topic_counts = df["nmf_topic"].value_counts().sort_index()
print("\n===== TOPIC PREVALENCE (document counts) =====")
for idx in range(N_TOPICS):
    cnt = topic_counts.get(idx, 0)
    pct = cnt / len(df) * 100
    print(f"Topic {idx} ({', '.join(nmf_topics[idx][:4])}): {cnt} docs ({pct:.1f}%)")

nmf_topic_summary = pd.DataFrame([
    {"topic_id": idx, "top_keywords": ", ".join(words), "doc_count": topic_counts.get(idx, 0),
     "pct_of_corpus": round(topic_counts.get(idx, 0)/len(df)*100, 2)}
    for idx, words in nmf_topics.items()
]).sort_values("doc_count", ascending=False)

nmf_topic_summary.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/nmf_topic_summary.csv", index=False)
df[["review_id", "product_type", "brand", "rating", "feedback_text",
    "nmf_topic", "nmf_topic_label", "nmf_topic_confidence"]].to_csv(
    "C:/Users/saksham singhal/Desktop/EV/files/data/nmf_document_topics.csv", index=False)

print("\nSaved: nmf_topic_summary.csv, nmf_document_topics.csv")

df.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_clean.csv", index=False)
