# Copyright (c) 2026 Abdullah Abuhassan <aabuhassan@unibz.it>
# Licensed under the MIT License — see LICENSE file for details.

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