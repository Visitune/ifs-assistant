# Analyser le JSON IFS
import json

with open('ifs_food_v8_fr.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Type: {type(data)}")

if isinstance(data, list):
    print(f"Nombre d'elements: {len(data)}")
    if len(data) > 0:
        print(f"Premier element: {data[0]}")
elif isinstance(data, dict):
    print(f"Cles: {list(data.keys())}")
