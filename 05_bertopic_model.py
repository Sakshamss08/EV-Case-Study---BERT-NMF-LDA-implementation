"""
STEP 5: BERTopic (Transformer-based Topic Modeling) -- REAL BERT VERSION
==========================================================================
Libraries: bertopic, sentence-transformers, umap-learn, hdbscan, scikit-learn
Pipeline:
  1. Embed each document with a pretrained sentence-transformer
     (all-MiniLM-L6-v2 -- true BERT-based semantic embeddings)
  2. Reduce dimensionality with UMAP
  3. Cluster embeddings with HDBSCAN (density-based -> finds natural
     number of topics AND flags outliers as topic -1)
  4. Represent each cluster with class-based TF-IDF (c-TF-IDF) to
     extract human-readable keywords per topic
Input:   lightly-cleaned text (clean_text_bert) to preserve semantic context
Output:  auto-discovered topics + keywords + document-topic assignment

RUN THIS ON A MACHINE WITH INTERNET ACCESS (e.g. your laptop, Colab, or any
environment that can reach huggingface.co). The first run will download the
'all-MiniLM-L6-v2' model (~90MB) and cache it locally under
~/.cache/torch/sentence_transformers/ -- subsequent runs are offline/fast.

Install requirements first:
    pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------
# Update this path to wherever your cleaned CSV lives locally
# (output of 02_preprocess.py -- must contain a 'clean_text_bert' column)
# ----------------------------------------------------------------------
INPUT_PATH = "C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_clean.csv"

df = pd.read_csv(INPUT_PATH)
texts = df["clean_text_bert"].tolist()

# ---- 1. Embedding model: real pretrained sentence-transformer ----
print("Loading sentence-transformer embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)  # (n_docs, 384)

# ---- 2. UMAP dimensionality reduction (deterministic via random_state) ----
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

# ---- 3. HDBSCAN clustering ----
hdbscan_model = HDBSCAN(
    min_cluster_size=25,     # min docs to form a topic (tune for your corpus size)
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# ---- 4. Vectorizer for topic keyword extraction (c-TF-IDF), with stopword removal ----
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3)

# ---- Assemble BERTopic pipeline ----
# NOTE: embedding_model IS passed here (unlike the offline fallback version)
# so BERTopic can also embed any new/unseen documents later via .transform()
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(texts, embeddings)

df["bert_topic"] = topics
df["bert_topic_confidence"] = [max(p) if hasattr(p, "__len__") else p for p in probs]

# ---- Topic info summary ----
topic_info = topic_model.get_topic_info()
print("\n===== BERTopic TOPICS (raw) =====")
print(topic_info.to_string())

# Reduce outliers (topic -1) by reassigning to nearest topic where possible
# HDBSCAN naturally discovers the topic count for you -- this is BERTopic's key
# advantage over fixed-K models like LDA/NMF: it does not force a pre-specified
# topic count, so granular customer concerns are not artificially blended together.
new_topics = topic_model.reduce_outliers(texts, topics, strategy="c-tf-idf")
topic_model.update_topics(texts, topics=new_topics, vectorizer_model=vectorizer_model)
df["bert_topic"] = new_topics

topic_info_after = topic_model.get_topic_info()
print("\n===== BERTopic TOPICS (after outlier reduction) =====")
print(topic_info_after.to_string())

# ---- (Optional) Consolidate to a target number of topics for business reporting ----
# Uncomment if you want a fixed comparable K (e.g. to line up with LDA/NMF at K=10):
# topic_model.reduce_topics(texts, nr_topics=10)
# df["bert_topic"] = topic_model.topics_
# topic_info_after = topic_model.get_topic_info()

# ---- Build readable labels ----
bert_topic_labels = {}
for row in topic_info_after.itertuples():
    tid = row.Topic
    words = topic_model.get_topic(tid)
    if words:
        top_words = [w for w, _ in words[:6]]
        bert_topic_labels[tid] = f"Topic {tid}: " + ", ".join(top_words)
    else:
        bert_topic_labels[tid] = f"Topic {tid}: (outlier/misc)"

df["bert_topic_label"] = df["bert_topic"].map(bert_topic_labels)

# ---- Save outputs ----
topic_info_after.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/bertopic_topic_summary.csv", index=False)
df[["review_id", "product_type", "brand", "rating", "feedback_text",
    "bert_topic", "bert_topic_label", "bert_topic_confidence"]].to_csv(
    "C:/Users/saksham singhal/Desktop/EV/files/data/bertopic_document_topics.csv", index=False)

df.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_clean.csv", index=False)

# ---- (Optional) Save the fitted model itself for reuse later ----
# topic_model.save("bertopic_model_ev", serialization="safetensors",
#                   save_ctfidf=True, save_embedding_model="all-MiniLM-L6-v2")

# ---- (Optional) Visualizations -- open these directly in a browser ----
# topic_model.visualize_topics().write_html("bertopic_intertopic_map.html")
# topic_model.visualize_barchart(top_n_topics=12).write_html("bertopic_barchart.html")
# topic_model.visualize_hierarchy().write_html("bertopic_hierarchy.html")

print("\nSaved: bertopic_topic_summary.csv, bertopic_document_topics.csv")
print("\nFinal topic count (excluding outlier bucket):", len(topic_info_after[topic_info_after.Topic != -1]))