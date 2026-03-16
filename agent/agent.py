# Copyright (c) 2026 Abdullah Abuhassan <aabuhassan@unibz.it>
# Licensed under the MIT License — see LICENSE file for details.

import pandas as pd
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# ── Load data ──
nlp_df = pd.read_csv("nlp/nlp_output.csv")
enriched_df = pd.read_csv("ml/activities_enriched.csv")

df = enriched_df.merge(
    nlp_df[["id", "summary", "keywords", "hazard_flags",
            "condition_flags", "has_hazard", "has_condition_info"]],
    on="id", how="left"
)

# ── Init Ollama with streaming ──
llm = ChatOllama(model="mistral", temperature=0.3, streaming=True)

# ── Build context snapshot ──
def build_context():
    total = len(df)
    open_trails = df[df["is_open"] == 1]
    hazard_trails = df[df["has_hazard"] == True]
    by_type = df[df["is_open"] == 1]["activity_type"].value_counts().head(5).to_dict()

    hazard_list = []
    for _, row in hazard_trails.head(5).iterrows():
        title = row.get("title") or "Unknown"
        hazards = row.get("hazard_flags", "")
        hazard_list.append(f"- {title}: {hazards}")

    return f"""
TRAILPULSESÜDTIROL — CURRENT SNAPSHOT
Total activities: {total}
Currently open: {len(open_trails)}
Hazard alerts: {len(hazard_trails)}

Open activities by type:
{json.dumps(by_type, indent=2)}

Hazard alerts (sample):
{chr(10).join(hazard_list) if hazard_list else "None detected"}
"""

# ── Build activity sample for queries ──
def build_activity_sample(activity_type=None):
    filtered = df[df["is_open"] == 1]
    if activity_type:
        filtered = filtered[filtered["activity_type"] == activity_type]
    return filtered.head(15)[
        ["title", "activity_type", "location",
         "distance_m", "difficulty", "summary"]
    ].to_string(index=False)

# ── Stream response to terminal ──
def stream_response(messages):
    print("\nTrailPulse: ", end="", flush=True)
    full = ""
    for chunk in llm.stream(messages):
        text = chunk.content
        print(text, end="", flush=True)
        full += text
    print("\n")
    return full

# ── Daily briefing ──
def generate_daily_briefing():
    context = build_context()
    messages = [
        SystemMessage(content="""You are TrailPulse, an intelligent trail conditions 
assistant for South Tyrol. Generate clear, concise daily briefings for outdoor 
enthusiasts based on real trail data. Be specific, mention actual numbers, 
and keep it under 150 words."""),
        HumanMessage(content=f"""Generate today's trail conditions briefing 
for South Tyrol based on this data:

{context}""")
    ]
    return stream_response(messages)

# ── Query handler ──
def answer_query(question):
    context = build_context()
    sample = build_activity_sample()

    messages = [
        SystemMessage(content="""You are TrailPulse, an intelligent trail assistant 
for South Tyrol. Answer questions about trails, activities, and conditions 
based on the provided data. Be specific, helpful, and concise. 
If the data doesn't contain the answer, say so clearly."""),
        HumanMessage(content=f"""Data snapshot:
{context}

Sample open activities:
{sample}

Question: {question}""")
    ]
    return stream_response(messages)

# ── Main ──
if __name__ == "__main__":
    print("=" * 55)
    print("   TRAILPULSESÜDTIROL — AI AGENT")
    print("=" * 55)
    print("Generating today's briefing...\n")

    generate_daily_briefing()

    print("=" * 55)
    print("Ask me anything about trails in South Tyrol.")
    print("Type 'exit' to quit.")
    print("=" * 55 + "\n")

    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            answer_query(question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break