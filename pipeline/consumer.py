import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "trail-status-changes",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

print("Listening for trail changes...")
for message in consumer:
    print(f"Received: {message.value}")