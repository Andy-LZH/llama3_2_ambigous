import json

def convert_polygons(data):
    for item in data:
        if "polygons" in item:
            # Convert each set of polygons to [[x, y], ...] format
            new_polygons = []
            for polygon_set in item["polygons"]:
                converted_set = []
                for polygon in polygon_set:
                    polygons_convert_set = []
                    for point in polygon:
                        polygons_convert_set.append(point["x"])
                        polygons_convert_set.append(point["y"])
                    converted_set.append(polygons_convert_set)
                new_polygons.append(converted_set)
            item["polygons"] = new_polygons
    return data


with open("answer_therapy_molmo_fs_raw.json", "r") as file:
    dataset = json.load(file)

converted_dataset = convert_polygons(dataset)

with open("answer_therapy_molmo_fs.json", "w") as file:
    json.dump(converted_dataset, file)
