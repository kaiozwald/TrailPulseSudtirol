import pandas as pd
import json
from langdetect import detect, LangDetectException

df = pd.read_csv("ml/activities_enriched.csv")
# ── Load raw API data for full text fields ──
# We need to re-fetch since CSV doesn't store the full Detail object
import requests

API_URL = "https://tourism.api.opendatahub.com/v1/ODHActivityPoi"
PAGE_SIZE = 200

def fetch_all():
    activities, page = [], 1
    while True:
        r = requests.get(API_URL, params={
            "tagfilter": "activity",
            "pagesize": PAGE_SIZE,
            "pagenumber": page
        }, timeout=10)
        items = r.json().get("Items", [])
        if not items:
            break
        activities.extend(items)
        page += 1
    return activities

def extract_texts(activity):
    detail = activity.get("Detail", {}) or {}
    
    def get_text(lang):
        lang_detail = detail.get(lang, {}) or {}
        title = lang_detail.get("Title", "") or ""
        desc = lang_detail.get("BaseText", "") or ""
        # fallback to other description fields
        if not desc:
            desc = lang_detail.get("AdditionalText", "") or ""
        return title.strip(), desc.strip()

    en_title, en_desc = get_text("en")
    de_title, de_desc = get_text("de")
    it_title, it_desc = get_text("it")

    # Detect language of the richest description
    combined = en_desc or de_desc or it_desc
    try:
        detected_lang = detect(combined) if combined else "unknown"
    except LangDetectException:
        detected_lang = "unknown"

    return {
        "id": activity.get("Id"),
        "en_title": en_title,
        "en_desc": en_desc,
        "de_title": de_title,
        "de_desc": de_desc,
        "it_title": it_title,
        "it_desc": it_desc,
        "detected_lang": detected_lang,
        "has_en_desc": bool(en_desc),
        "has_de_desc": bool(de_desc),
        "has_it_desc": bool(it_desc),
        "desc_length": len(combined)
    }

if __name__ == "__main__":
    print("Fetching full dataset...")
    activities = fetch_all()
    
    print("Extracting text fields...")
    records = [extract_texts(a) for a in activities]
    texts_df = pd.DataFrame(records)
    
    print(f"\nTotal records: {len(texts_df)}")
    print(f"\nLanguage coverage:")
    print(f"  Has English description:  {texts_df['has_en_desc'].sum()}")
    print(f"  Has German description:   {texts_df['has_de_desc'].sum()}")
    print(f"  Has Italian description:  {texts_df['has_it_desc'].sum()}")
    print(f"\nDetected languages:")
    print(texts_df["detected_lang"].value_counts().head(8))
    print(f"\nDescription length stats:")
    print(texts_df[texts_df["desc_length"] > 0]["desc_length"].describe())
    
    texts_df.to_csv("nlp/texts.csv", index=False)
    print("\nSaved to nlp/texts.csv")