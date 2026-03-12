import pandas as pd
import spacy
import re
from collections import Counter

# ── Load spacy models ──
nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")

texts_df = pd.read_csv("nlp/texts.csv")
activities_df = pd.read_csv("ml/activities_enriched.csv")

# ── Merge on id ──
df = activities_df.merge(texts_df, on="id", how="left")

IMPUTED_DISTANCE = 900.0
IMPUTED_DURATION = 4.2
IMPUTED_ALTITUDE = 183.0

# ════════════════════════════════
# PART A — Keyword Extraction
# ════════════════════════════════

def extract_keywords(text, lang="en", top_n=5):
    """Extract keywords using spaCy NER and noun chunks."""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return []
    try:
        model = nlp_de if lang == "de" else nlp_en
        doc = model(text[:1000])  # limit for speed
        
        # Extract noun chunks and named entities
        candidates = []
        for chunk in doc.noun_chunks:
            clean = chunk.text.strip().lower()
            if 2 < len(clean) < 40:
                candidates.append(clean)
        for ent in doc.ents:
            clean = ent.text.strip().lower()
            if 2 < len(clean) < 40:
                candidates.append(clean)

        # Count and return top N
        counts = Counter(candidates)
        return [kw for kw, _ in counts.most_common(top_n)]
    except Exception:
        return []

# ════════════════════════════════
# PART B — Description Summaries
# ════════════════════════════════

def summarise_from_description(row):
    """Generate summary from actual description text."""
    desc = row.get("en_desc") or row.get("de_desc") or row.get("it_desc") or ""
    if not isinstance(desc, str) or len(desc) < 20:
        return None
    # Clean HTML tags if present
    desc = re.sub(r"<[^>]+>", " ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    # Return first 2 sentences as summary
    sentences = re.split(r"(?<=[.!?])\s+", desc)
    return " ".join(sentences[:2])

def generate_template_summary(row):
    """Generate structured summary from structured fields."""
    title = row.get("en_title") or row.get("de_title") or row.get("it_title") or "Unknown trail"
    activity_type = str(row.get("activity_type", "activity")).capitalize()
    location = row.get("location", "South Tyrol")
    distance = row.get("distance_m")
    duration = row.get("duration_min")
    altitude = row.get("altitude_diff")
    difficulty_map = {2: "easy", 4: "moderate", 6: "challenging"}
    difficulty = difficulty_map.get(row.get("difficulty"), "scenic")
    has_rentals = row.get("has_rentals", 0)

    parts = [f"{activity_type} route in {location}."]

    # Only include metrics if they are not imputed median values
    if distance and distance > 0 and distance != IMPUTED_DISTANCE:
        parts.append(f"Distance: {int(distance)}m.")
    if duration and duration > 0 and duration != IMPUTED_DURATION:
        parts.append(f"Duration: approx. {int(duration)} min.")
    if altitude and altitude > 0 and altitude != IMPUTED_ALTITUDE:
        parts.append(f"Elevation gain: {int(altitude)}m ({difficulty} difficulty).")
    if has_rentals:
        parts.append("Rental equipment available.")

    return " ".join(parts)

# ════════════════════════════════
# PART C — Process full dataset
# ════════════════════════════════

print("Processing dataset...")
summaries = []
keywords_list = []

for _, row in df.iterrows():
    # Summary
    desc_summary = summarise_from_description(row)
    if desc_summary:
        summary = desc_summary
        source = "description"
    else:
        summary = generate_template_summary(row)
        source = "template"

    # Keywords
    desc = row.get("en_desc") or row.get("de_desc") or ""
    lang = "de" if (not row.get("en_desc") and row.get("de_desc")) else "en"
    keywords = extract_keywords(desc, lang=lang)

    # Fallback keywords from title if no description
    if not keywords:
        title = row.get("en_title") or row.get("de_title") or ""
        keywords = extract_keywords(title, lang="en")

    summaries.append(summary)
    keywords_list.append(", ".join(keywords) if keywords else "")

df["summary"] = summaries
df["keywords"] = keywords_list
df["summary_source"] = ["description" if s else "template"
                         for s in [summarise_from_description(r)
                         for _, r in df.iterrows()]]

# ── Save ──
output = df[["id", "en_title", "activity_type", "location",
             "difficulty", "distance_m", "duration_min",
             "summary", "keywords", "summary_source"]]
output.to_csv("nlp/summaries.csv", index=False)

print(f"Done — {len(output)} records processed")
print(f"\nSummary sources:")
print(output["summary_source"].value_counts())
print(f"\nRecords with keywords: {(output['keywords'] != '').sum()}")
print(f"\nSample description-based summary:")
desc_sample = output[output["summary_source"] == "description"].iloc[0]
print(f"  Title: {desc_sample['en_title']}")
print(f"  Summary: {desc_sample['summary']}")
print(f"  Keywords: {desc_sample['keywords']}")
print(f"\nSample template-based summary:")
tmpl_sample = output[output["summary_source"] == "template"].iloc[0]
print(f"  Title: {tmpl_sample['en_title']}")
print(f"  Summary: {tmpl_sample['summary']}")