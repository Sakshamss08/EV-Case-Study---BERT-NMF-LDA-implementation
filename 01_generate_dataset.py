"""
DATA GENERATION
Simulates a realistic customer-feedback corpus for a company evaluating
whether to launch an EV Car or EV Scooter.

Each row = one piece of customer feedback:
  review_id, source, product_type (car/scooter), competitor_brand,
  rating, region, date, feedback_text
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

themes = {
    "Range Anxiety": [
        "The real world range is much lower than advertised, I only get {n} km on a full charge.",
        "Range drops badly in winter, I lost almost 30% range in cold weather.",
        "I'm always worried about running out of charge before reaching my destination.",
        "Claimed range of {n} km is misleading, actual range is way less in city traffic.",
        "Range anxiety is real, I plan every trip around charging stops now.",
        "AC usage kills the range, on hot days I barely get half the claimed distance.",
        "For daily commute the range is fine but long trips are stressful because of low range.",
        "Battery range degrades faster than expected after just one year of use.",
    ],
    "Charging Infrastructure": [
        "Finding a working charging station in my city is still very difficult.",
        "Public chargers are often broken or occupied, need more charging stations urgently.",
        "Home charging setup was expensive and took weeks to install.",
        "Fast charging network needs serious expansion on highways.",
        "Charging takes too long, almost 6 hours for a full charge at home.",
        "Love the app that shows nearby charging stations, very convenient feature.",
        "Highway charging stops are rare, long distance travel is a nightmare.",
        "Charging infrastructure in tier 2 and tier 3 cities is almost non-existent.",
    ],
    "Battery Life & Degradation": [
        "Worried about battery degradation and replacement cost after warranty ends.",
        "Battery health dropped to 85% within 18 months of daily use.",
        "The battery warranty of {n} years gives me confidence in long term reliability.",
        "Battery replacement cost is almost half the price of the vehicle, very concerning.",
        "Battery performance is stable even after two years of heavy usage.",
        "No proper information on how battery degrades over time, company should be transparent.",
    ],
    "Price & Subsidy": [
        "The upfront price is high but government subsidy makes it affordable.",
        "Running cost is much lower than petrol, saving a lot on fuel every month.",
        "Total cost of ownership is great once you factor in low maintenance and fuel savings.",
        "Price after subsidy is still higher than a comparable petrol scooter.",
        "EV subsidy scheme in my state got delayed and the discount amount reduced suddenly.",
        "Resale value of electric vehicles is still uncertain, worried about depreciation.",
        "Insurance premium for EV is surprisingly high compared to petrol vehicles.",
        "For the price, I expected better build quality and more features.",
    ],
    "Service & After-Sales": [
        "Service center staff are not well trained to handle EV specific issues.",
        "There are very few authorized service centers, had to travel {n} km for repair.",
        "After sales support has been excellent, quick response and doorstep service.",
        "Spare parts availability is a major issue, waited a month for a replacement part.",
        "Customer support over phone is helpful and resolves issues quickly.",
        "Waiting time for service appointment is too long, more service centers needed.",
        "Had a great experience with the service team, they explained everything clearly.",
    ],
    "Performance & Ride Quality": [
        "Acceleration is smooth and instant torque makes overtaking effortless.",
        "Handling and stability at high speed feels very confident and safe.",
        "Suspension is too stiff, ride quality is uncomfortable on bad roads.",
        "The motor is powerful and pickup is much better than petrol variants.",
        "Braking system feels sharp and responsive, regenerative braking works well.",
        "Top speed is decent for city use but not enough for highway cruising.",
        "Noise levels are impressively low, riding experience is very smooth and quiet.",
        "Build quality feels premium and the ride is extremely comfortable.",
    ],
    "Design & Aesthetics": [
        "The design looks futuristic and attracts a lot of attention on the road.",
        "Interior quality feels cheap for the price point, needs improvement.",
        "Loved the color options and the overall styling of the vehicle.",
        "Storage space under the seat is very limited, not practical for daily use.",
        "Compact size makes it easy to park and maneuver in city traffic.",
        "Build materials feel sturdy but the design is a bit too plain compared to rivals.",
        "LED lighting and digital dashboard give it a premium, modern look.",
    ],
    "Software & Connectivity": [
        "The companion app keeps crashing, needs urgent software updates.",
        "Over the air software updates are a great feature, keeps improving over time.",
        "Touchscreen infotainment system lags and is not very intuitive to use.",
        "Bluetooth connectivity drops frequently while riding, very frustrating.",
        "Navigation system integrated with range prediction is genuinely useful.",
        "Software glitches caused the dashboard to freeze twice in a month.",
        "Smart key and app based lock features work flawlessly, very convenient.",
    ],
    "Safety Features": [
        "Advanced safety features like ABS and traction control give peace of mind.",
        "Read reports about battery fires, safety is my biggest concern before buying.",
        "The vehicle has good crash test ratings which reassured my purchase decision.",
        "Emergency braking and collision alert systems work really well in traffic.",
        "Heard about a thermal runaway incident, this has shaken my trust in EV safety.",
        "Stability control on wet roads is impressive and inspires confidence.",
    ],
    "Environmental Impact": [
        "Switching to EV feels good knowing I'm reducing my carbon footprint.",
        "Zero tailpipe emissions is the main reason I chose an electric vehicle.",
        "Concerned about environmental impact of battery disposal and recycling.",
        "Contributing to a cleaner city is a big motivation behind going electric.",
        "Energy used to charge still comes mostly from coal in my region, defeats the purpose somewhat.",
    ],
}

product_types = ["EV Car", "EV Scooter"]
sources = ["Survey", "Social Media", "Customer Support", "Marketplace Review"]
car_brands = ["Tata Nexon EV", "MG ZS EV", "Hyundai Kona", "Mahindra XUV400", "BYD Atto 3", "Tesla Model 3"]
scooter_brands = ["Ola S1", "Ather 450X", "TVS iQube", "Bajaj Chetak", "Ampere Magnus", "Hero Vida V1"]
regions = ["North", "South", "East", "West", "Central"]

rows = []
review_id = 1000

theme_names = list(themes.keys())

n_records = 1200  # decent corpus size for topic modeling

start_date = datetime(2024, 1, 1)

for i in range(n_records):
    product = random.choice(product_types)
    brand = random.choice(car_brands) if product == "EV Car" else random.choice(scooter_brands)
    source = random.choice(sources)
    region = random.choice(regions)
    date = start_date + timedelta(days=random.randint(0, 545))

    # Each feedback mentions 1-3 themes (realistic: people often mix concerns)
    num_themes = np.random.choice([1, 2, 3], p=[0.55, 0.35, 0.10])
    chosen_themes = random.sample(theme_names, num_themes)

    sentences = []
    for t in chosen_themes:
        template = random.choice(themes[t])
        if "{n}" in template:
            n_val = random.choice([80, 100, 120, 150, 200, 250, 300, 15, 20, 30, 3, 5, 8])
            template = template.format(n=n_val)
        sentences.append(template)

    text = " ".join(sentences)


    negative_markers = ["worried", "difficult", "nightmare", "cheap", "frustrating", "shaken",
                         "delayed", "reduced", "uncomfortable", "issue", "waited", "concerning",
                         "crashing", "drops", "glitches", "high", "uncertain", "lower"]
    neg_count = sum(1 for m in negative_markers if m in text.lower())
    base_rating = 4.2 - 0.35 * neg_count + np.random.normal(0, 0.6)
    rating = int(np.clip(round(base_rating), 1, 5))

    rows.append({
        "review_id": f"EV{review_id}",
        "source": source,
        "product_type": product,
        "brand": brand,
        "region": region,
        "date": date.strftime("%Y-%m-%d"),
        "rating": rating,
        "feedback_text": text,
        "_true_themes": ", ".join(chosen_themes)  # kept for validation only, dropped before modeling
    })
    review_id += 1

df = pd.DataFrame(rows)
df.to_csv("C:/Users/saksham singhal/Desktop/EV/files/data/ev_customer_feedback_raw.csv", index=False)
print("Dataset shape:", df.shape)
print(df.head(3).to_string())
print("\nProduct type distribution:\n", df.product_type.value_counts())
print("\nSource distribution:\n", df.source.value_counts())
