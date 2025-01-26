import json
import random

dataset = json.load(open("MSRA.json"))
dataset_alter = json.load(open("MSRA_molmo_raw.json"))

for i in range(len(dataset)):
    dataset_alter[i]["masks"] = dataset[i]["mask"]

with open("MSRA_molmo.json", "w") as f:
    json.dump(dataset_alter, f)