import pandas as pd
import re

df = pd.read_csv("nlp/summaries.csv")
texts_df = pd.read_csv("nlp/texts.csv")
df = df.merge(texts_df[["id", "en_desc", "de_desc", "it_desc"]], on="id", how="left")

# ── Hazard and condition keyword dictionaries ──
HAZARD_PATTERNS = {
    "closed":       r"\b(closed|out of order|gesperrt|chiuso|not accessible)\b",
    "wind":         r"\b(wind|storm|sturm|vento|gale)\b",
    "ice":          r"\b(ice|icy|verglas|ghiaccio|glatt)\b",
    "snow":         r"\b(snow|schnee|neve|avalanche|lawine)\b",
    "flood":        r"\b(flood|flooding|hochwasser|alluvione)\b",
    "maintenance":  r"\b(maintenance|repair|instandhaltung|manutenzione|works)\b",
    "dangerous":    r"\b(danger|dangerous|gefahr|pericolo|hazard|risiko)\b",
    "seasonal":     r"\b(seasonal|only in summer|only in winter|nur im sommer|nur im winter)\b",
}

CONDITION_PATTERNS = {
    "groomed":      r"\b(groomed|prepared|präpariert|preparato|loipe)\b",
    "scenic":       r"\b(scenic|panoramic|aussicht|panorama|view|vista)\b",
    "family":       r"\b(family|children|kids|familien|bambini|kinder)\b",
    "rental":       r"\b(rental|hire|verleih|noleggio|equipment)\b",
    "guided":       r"\b(guided|guide|führer|guida|accompanied)\b",
}

def detect_flags(row):
    text = " ".join(filter(None, [
        str(row.get("en_desc", "") or ""),
        str(row.get("de_desc", "") or ""),
        str(row.get("it_desc", "") or ""),
        str(row.get("summary", "") or ""),
    ])).lower()

    hazards = [k for k, pattern in HAZARD_PATTERNS.items()
               if re.search(pattern, text, re.IGNORECASE)]
    conditions = [k for k, pattern in CONDITION_PATTERNS.items()
                  if re.search(pattern, text, re.IGNORECASE)]

    return {
        "hazard_flags": ", ".join(hazards) if hazards else "",
        "condition_flags": ", ".join(conditions) if conditions else "",
        "has_hazard": bool(hazards),
        "has_condition_info": bool(conditions)
    }

print("Detecting hazards and conditions...")
flags = pd.DataFrame([detect_flags(row) for _, row in df.iterrows()])
df = pd.concat([df, flags], axis=1)

# ── Save final NLP output ──
output = df[["id", "en_title", "activity_type", "location",
             "difficulty", "distance_m", "duration_min",
             "summary", "keywords", "summary_source",
             "hazard_flags", "condition_flags",
             "has_hazard", "has_condition_info"]]
output.to_csv("nlp/nlp_output.csv", index=False)

print(f"Done — {len(output)} records")
print(f"\nRecords with hazard flags:    {output['has_hazard'].sum()}")
print(f"Records with condition info:  {output['has_condition_info'].sum()}")
print(f"\nHazard type breakdown:")
all_hazards = ", ".join(output["hazard_flags"].dropna())
for h in HAZARD_PATTERNS.keys():
    count = all_hazards.count(h)
    if count > 0:
        print(f"  {h}: {count}")
print(f"\nSample hazard record:")
hazard_sample = output[output["has_hazard"]].iloc[0]
print(f"  Title:    {hazard_sample['en_title']}")
print(f"  Summary:  {hazard_sample['summary']}")
print(f"  Hazards:  {hazard_sample['hazard_flags']}")
print(f"\nSample condition record:")
cond_sample = output[output["has_condition_info"]].iloc[0]
print(f"  Title:     {cond_sample['en_title']}")
print(f"  Summary:   {cond_sample['summary']}")
print(f"  Conditions: {cond_sample['condition_flags']}")