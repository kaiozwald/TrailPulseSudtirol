# Copyright (c) 2026 Abdullah Abuhassan <aabuhassan@unibz.it>
# Licensed under the MIT License — see LICENSE file for details.
 
import json
import time
import random
import logging
import requests
from kafka import KafkaProducer

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline/simulator.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

API_URL = "https://tourism.api.opendatahub.com/v1/ODHActivityPoi"
TOPIC = "trail-status-changes"
SIMULATE_INTERVAL = 15  # seconds between simulated events
CHANGES_PER_CYCLE = 3   # how many trails "change" per cycle

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def fetch_sample(size=50):
    params = {"tagfilter": "activity", "pagesize": size, "pagenumber": 1}
    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json().get("Items", [])

def simulate_change(activity):
    previous_open = random.choice([True, False])
    current_open = not previous_open  # flip it

    return {
        "id": activity.get("Id"),
        "title": activity.get("Detail", {}).get("en", {}).get("Title", "Unknown"),
        "previous": {
            "IsOpen": previous_open,
            "IsPrepared": random.choice([True, False]),
            "LastUpdate": "2026-03-11T08:00:00"
        },
        "current": {
            "IsOpen": current_open,
            "IsPrepared": random.choice([True, False]),
            "LastUpdate": "2026-03-12T08:00:00"
        },
        "location": activity.get("LocationInfo", {}).get("MunicipalityInfo", {}).get("Name", {}).get("en", "Unknown"),
        "activity_type": activity.get("Tags", [{}])[0].get("Id", "unknown") if activity.get("Tags") else "unknown",
        "simulated": True  # flag so you always know this is synthetic
    }

def run():
    log.info("TrailPulseSüdtirol simulator started — fetching activity sample...")
    activities = fetch_sample(50)
    log.info(f"Loaded {len(activities)} activities for simulation pool")

    while True:
        sample = random.sample(activities, min(CHANGES_PER_CYCLE, len(activities)))
        for activity in sample:
            event = simulate_change(activity)
            producer.send(TOPIC, event)
            status = "OPENED" if event["current"]["IsOpen"] else "CLOSED"
            log.info(f"[SIMULATED] {status}: {event['title']} | {event['location']} | type: {event['activity_type']}")
        log.info(f"Simulation cycle done — {len(sample)} events emitted")
        time.sleep(SIMULATE_INTERVAL)

if __name__ == "__main__":
    run()