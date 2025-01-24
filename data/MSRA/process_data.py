import json
import random

dataset = json.load(open("MSRA_RLE_627.json"))

matched_data = []

for key in dataset.keys():
    matched_data.append(
        {
            "imageURL": "data/MSRA/MSRA_Image/" + key,
            "question": dataset[key]["question"],
            "dataset": "MSRA",
            "ambiguity": "None",
            "mask": dataset[key]["RLE"],
        }
    )

# saved as MSRA.json
with open("MSRA.json", "w") as f:
    json.dump(matched_data, f)

# save also a small subset for visualization
subset = random.sample(matched_data, 10)
with open("MSRA_subset.json", "w") as f:
    json.dump(subset, f)
