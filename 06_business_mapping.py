"""
STEP 6: BUSINESS-THEME MAPPING & CROSS-MODEL COMPARISON
============================================================
Raw topic-model output (numbered topics + keyword lists) is not directly
usable by business stakeholders. This step maps each model's topics to
the SAME set of human-readable business themes so leadership can compare
"what LDA found" vs "what NMF found" vs "what BERTopic found" side by side,
and see where they agree (signal) vs disagree (needs deeper investigation).

Business themes (derived by inspecting keyword lists from all 3 models):
  1. Range Anxiety
  2. Charging Infrastructure
  3. Battery Life & Degradation
  4. Price & Subsidy / Cost of Ownership
  5. Service & After-Sales
  6. Performance & Ride Quality
  7. Design & Build Quality
  8. Software & Connectivity
  9. Safety Features
  10. Environmental Impact
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_clean.csv")

# ---------------------------------------------------------------
# Manual keyword-based mapping (built by inspecting each model's
# top-keyword output from steps 3-5). This is the standard real-world
# practice: an analyst reviews auto-generated topic keywords and assigns
# a business label -- topic models surface structure, humans name it.
# ---------------------------------------------------------------

lda_map = {
    0: "Price & Subsidy / Cost of Ownership",   # cost, replacement cost, charging station
    1: "Software & Connectivity",                # navigation system, integrated range prediction
    2: "Design & Build Quality",                 # feel, quality, build quality
    3: "Safety Features",                        # safety, advanced safety, peace mind
    4: "Software & Connectivity",                # software, update, glitch
    5: "Safety Features",                        # service center, safety, fire report
    6: "Battery Life & Degradation",              # charging, battery, health dropped
    7: "Performance & Ride Quality",              # braking, regenerative braking
    8: "Environmental Impact",                    # energy, coal, region defeat purpose
    9: "Environmental Impact",                    # price, subsidy, disposal, recycling
}

nmf_map = {
    0: "Battery Life & Degradation",              # cost, replacement cost, degradation
    1: "Environmental Impact",                    # coal region, purpose somewhat
    2: "Performance & Ride Quality",               # braking, regenerative braking
    3: "Safety Features",                         # abs, traction control, advanced safety
    4: "Battery Life & Degradation",              # daily use, battery health dropped
    5: "Environmental Impact",                    # battery disposal, recycling
    6: "Software & Connectivity",                 # app, software update, crashing
    7: "Performance & Ride Quality",               # riding, smooth, noise level
    8: "Service & After-Sales",                   # service center, waiting time
    9: "Software & Connectivity",                 # dashboard, glitch
}

bert_map = {
    0: "Performance & Ride Quality",     # low riding experience, quiet
    1: "Software & Connectivity",        # smart key, lock features
    2: "Software & Connectivity",        # dashboard, software glitches
    3: "Environmental Impact",           # zero tailpipe emissions
    4: "Environmental Impact",           # ev carbon footprint
    5: "Charging Infrastructure",        # highway, price, use highway
    6: "Environmental Impact",           # battery disposal, recycling
    7: "Environmental Impact",           # cleaner city, motivation
    8: "Service & After-Sales",          # sales support, excellent quick
    9: "Design & Build Quality",         # charging, build quality, premium
    10: "Environmental Impact",          # energy, coal region
    11: "Safety Features",               # alert systems, emergency braking
    12: "Charging Infrastructure",       # charge home, charging takes long
    13: "Price & Subsidy / Cost of Ownership",  # government subsidy, affordable
    14: "Performance & Ride Quality",    # braking, sharp responsive
    15: "Performance & Ride Quality",    # acceleration, instant torque
    16: "Range Anxiety",                 # range, roads, battery, stability
    17: "Safety Features",               # crash test, purchase decision
    18: "Software & Connectivity",       # software updates
    19: "Price & Subsidy / Cost of Ownership",  # subsidy, intuitive use
    20: "Safety Features",               # advanced safety, abs traction
    21: "Battery Life & Degradation",    # company transparent, battery degrades
    22: "Charging Infrastructure",       # nearby charging
    23: "Service & After-Sales",         # service centers, time
}

df["lda_business_theme"] = df["lda_topic"].map(lda_map)
df["nmf_business_theme"] = df["nmf_topic"].map(nmf_map)
df["bert_business_theme"] = df["bert_topic"].map(bert_map)

# ---------------------------------------------------------------
# Cross-model agreement: for how many documents do all 3 models
# agree on the business theme? This is a strong validation signal.
# ---------------------------------------------------------------
df["models_agree_all3"] = (
    (df["lda_business_theme"] == df["nmf_business_theme"]) &
    (df["nmf_business_theme"] == df["bert_business_theme"])
)
df["models_agree_2of3"] = (
    (df["lda_business_theme"] == df["nmf_business_theme"]) |
    (df["nmf_business_theme"] == df["bert_business_theme"]) |
    (df["lda_business_theme"] == df["bert_business_theme"])
)

agreement_rate_all3 = df["models_agree_all3"].mean() * 100
agreement_rate_2of3 = df["models_agree_2of3"].mean() * 100
print(f"All 3 models agree on business theme: {agreement_rate_all3:.1f}% of documents")
print(f"At least 2 of 3 models agree: {agreement_rate_2of3:.1f}% of documents")

# ---------------------------------------------------------------
# THE CORE DELIVERABLE: "What truly matters to customers" ranking
# Aggregate theme prevalence across all 3 models (average rank/share)
# ---------------------------------------------------------------
theme_prevalence = pd.DataFrame({
    "LDA_pct": df["lda_business_theme"].value_counts(normalize=True) * 100,
    "NMF_pct": df["nmf_business_theme"].value_counts(normalize=True) * 100,
    "BERTopic_pct": df["bert_business_theme"].value_counts(normalize=True) * 100,
}).fillna(0)

theme_prevalence["Average_pct"] = theme_prevalence[["LDA_pct", "NMF_pct", "BERTopic_pct"]].mean(axis=1)
theme_prevalence = theme_prevalence.sort_values("Average_pct", ascending=False)
theme_prevalence = theme_prevalence.round(2)

print("\n===== WHAT TRULY MATTERS TO CUSTOMERS (aggregated across 3 models) =====")
print(theme_prevalence.to_string())

# ---------------------------------------------------------------
# Cut by product type: does what matters differ for Car vs Scooter buyers?
# ---------------------------------------------------------------
theme_by_product = pd.crosstab(df["bert_business_theme"], df["product_type"], normalize="columns") * 100
theme_by_product = theme_by_product.round(2).sort_values("EV Car", ascending=False)
print("\n===== THEME PREVALENCE BY PRODUCT TYPE (BERTopic-based, %) =====")
print(theme_by_product.to_string())

# ---------------------------------------------------------------
# Cut by rating: which themes drive LOW ratings (pain points to fix
# before launch) vs HIGH ratings (strengths to market)?
# ---------------------------------------------------------------
avg_rating_by_theme = df.groupby("bert_business_theme")["rating"].mean().sort_values()
theme_volume = df["bert_business_theme"].value_counts()

pain_vs_strength = pd.DataFrame({
    "avg_rating": avg_rating_by_theme,
    "doc_count": theme_volume
}).sort_values("avg_rating")
pain_vs_strength["avg_rating"] = pain_vs_strength["avg_rating"].round(2)

print("\n===== PAIN POINTS (low avg rating) vs STRENGTHS (high avg rating) =====")
print(pain_vs_strength.to_string())

# Save everything
df.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_final.csv", index=False)
theme_prevalence.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/theme_prevalence_cross_model.csv")
theme_by_product.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/theme_by_product_type.csv")
pain_vs_strength.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/pain_points_vs_strengths.csv")

print("\nSaved: ev_customer_feedback_final.csv, theme_prevalence_cross_model.csv, "
      "theme_by_product_type.csv, pain_points_vs_strengths.csv")
