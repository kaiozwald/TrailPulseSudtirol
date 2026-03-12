import os
import json
import time
import logging
import requests
from kafka import KafkaProducer

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("poller.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

API_URL = "https://tourism.api.opendatahub.com/v1/ODHActivityPoi"
TOPIC = "trail-status-changes"
POLL_INTERVAL = 60  # seconds
PAGE_SIZE = 200
STATE_FILE = "state.json"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

last_state = {}



def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            log.info("Loaded previous state from disk")
            return json.load(f)
    log.info("No previous state found — starting fresh")
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def fetch_all_activities():
    activities = []
    page = 1
    while True:
        params = {"tagfilter": "activity", "pagesize": PAGE_SIZE, "pagenumber": page}
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get("Items", [])
        if not items:
            break
        activities.extend(items)
        log.info(f"Fetched page {page} — {len(items)} items")
        page += 1
    return activities

def check_changes(activities):
    changes = []
    for activity in activities:
        aid = activity.get("Id")
        current = {
            "IsOpen": activity.get("IsOpen"),
            "IsPrepared": activity.get("IsPrepared"),
            "LastUpdate": activity.get("_Meta", {}).get("LastUpdate")
        }
        if aid in last_state and last_state[aid] != current:
            changes.append({
                "id": aid,
                "title": activity.get("Detail", {}).get("en", {}).get("Title", "Unknown"),
                "previous": last_state[aid],
                "current": current,
                "location": activity.get("LocationInfo", {}).get("MunicipalityInfo", {}).get("Name", {}).get("en", "Unknown"),
                "activity_type": activity.get("Tags", [{}])[0].get("Id", "unknown") if activity.get("Tags") else "unknown"
            })
        last_state[aid] = current
    return changes

def run():
    global last_state
    last_state = load_state()
    log.info("TrailPulseSüdtirol poller started — full dataset mode")
    while True:
        try:
            activities = fetch_all_activities()
            changes = check_changes(activities)
            for change in changes:
                producer.send(TOPIC, change)
                log.info(f"Event emitted: {change['title']} | Location: {change['location']} | IsOpen: {change['current']['IsOpen']}")
            save_state(last_state)
            log.info(f"Cycle complete — {len(activities)} activities polled, {len(changes)} changes emitted")
        except Exception as e:
            log.error(f"Polling error: {e}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    run()
