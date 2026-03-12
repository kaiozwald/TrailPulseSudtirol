import requests
import pandas as pd

API_URL = "https://tourism.api.opendatahub.com/v1/ODHActivityPoi"
PAGE_SIZE = 200

def fetch_all():
    activities, page = [], 1
    while True:
        r = requests.get(API_URL, params={"tagfilter": "activity", "pagesize": PAGE_SIZE, "pagenumber": page}, timeout=10)
        items = r.json().get("Items", [])
        if not items:
            break
        activities.extend(items)
        page += 1
    return activities

def extract_features(activity):
    detail = activity.get("Detail", {}).get("en", {})
    gps = activity.get("GpsInfo", [{}])[0] if activity.get("GpsInfo") else {}
    location_info = activity.get("LocationInfo") or {}
    municipality = location_info.get("MunicipalityInfo") or {}
    name = municipality.get("Name") or {}
    location = name.get("en", "Unknown")
    tags = [t.get("Id", "") for t in activity.get("Tags", [])]

    return {
        "id": activity.get("Id"),
        "title": detail.get("Title", "Unknown"),
        "difficulty": activity.get("Difficulty"),
        "distance_m": activity.get("DistanceLength"),
        "duration_min": activity.get("DistanceDuration"),
        "altitude_diff": activity.get("AltitudeDifference"),
        "altitude_start": gps.get("Altitude"),
        "has_rentals": int(bool(activity.get("HasRentals"))),
        "lift_available": int(bool(activity.get("LiftAvailable"))),
        "is_open": int(bool(activity.get("IsOpen"))),
        "is_prepared": int(bool(activity.get("IsPrepared"))),
        "location": location,
        "activity_type": tags[0] if tags else "unknown",
    }

if __name__ == "__main__":
    print("Fetching full dataset...")
    activities = fetch_all()
    df = pd.DataFrame([extract_features(a) for a in activities])

    print(f"\nTotal records: {len(df)}")
    print(f"\nColumn overview:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nActivity types:")
    print(df["activity_type"].value_counts().head(10))
    print(f"\nDifficulty distribution:")
    print(df["difficulty"].value_counts())
    print(f"\nNumerical summary:")
    print(df[["distance_m", "duration_min", "altitude_diff", "altitude_start"]].describe())

    df.to_csv("ml/activities.csv", index=False)
    print("\nDataset saved to ml/activities.csv")