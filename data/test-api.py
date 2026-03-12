import requests
import json

URL = "https://tourism.api.opendatahub.com/v1/ODHActivityPoi?tagfilter=activity&pagesize=5"

def main():
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()

        data = response.json()

        records = data.get("Items", [])

        print(f"Retrieved {len(records)} activity records:\n")

        for i, item in enumerate(records, start=1):
            name = item.get("Detail", {}).get("en", {}).get("Title") \
                   or item.get("Detail", {}).get("de", {}).get("Title") \
                   or item.get("Id", "Unknown")

            print(f"{i}. {name}")
            print(json.dumps(item, indent=2))
            print("-" * 60)

    except requests.RequestException as e:
        print("API request failed:", e)


if __name__ == "__main__":
    main()
