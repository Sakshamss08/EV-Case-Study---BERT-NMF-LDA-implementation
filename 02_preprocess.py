"""
STEP 2: TEXT PREPROCESSING
============================
Two preprocessing tracks are needed because LDA/NMF and BERTopic want
different inputs:

  Track A (for LDA & NMF - classical bag-of-words models):
      lowercase -> remove punctuation/numbers -> tokenize ->
      remove stopwords -> lemmatize -> rejoin to cleaned string
      (these models rely on word co-occurrence counts, so heavy
       cleaning improves topic coherence)

  Track B (for BERTopic - transformer embeddings):
      light cleaning only (keep casing/punctuation/context mostly intact)
      because BERT-based embeddings use full sentence context;
      over-cleaning HURTS embedding quality.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

df = pd.read_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_raw.csv")

stop_words = set(stopwords.words('english'))
# domain-specific additions: generic words that add noise, not signal
custom_stopwords = {"ev", "vehicle", "vehicles", "electric", "get", "really", "much", "also",
                     "one", "even", "still", "im", "ive", "would", "could", "like", "good", "bad"}
stop_words = stop_words.union(custom_stopwords)

lemmatizer = WordNetLemmatizer()

def clean_for_classical_models(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # remove numbers & punctuation
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def clean_for_bertopic(text):
    # light touch: normalize whitespace only, preserve sentence structure
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text_classical"] = df["feedback_text"].apply(clean_for_classical_models)
df["clean_text_bert"] = df["feedback_text"].apply(clean_for_bertopic)

# Drop rows that became empty after cleaning (rare, but safe to check)
before = len(df)
df = df[df["clean_text_classical"].str.strip() != ""]
after = len(df)
print(f"Dropped {before-after} empty rows after cleaning")

df.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_clean.csv", index=False)

print("Sample cleaned text (classical):")
for t in df["clean_text_classical"].head(5):
    print(" -", t)

print("\nSample cleaned text (BERT):")
for t in df["clean_text_bert"].head(5):
    print(" -", t)

print("\nFinal shape:", df.shape)
