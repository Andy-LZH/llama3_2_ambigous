import json
import pandas as pd

# load data from json file
with open("test.json") as f:
    data = json.load(f)

# print only ambigous data
data = data["ambigous"]

# now mark this into a dataframe with manually selected columns
# and save it into a csv file

# columns only include id, question, selected_parts_polygon, selected_parts_text, selected_objects_polygon, selected_objects_text

# itearting over the data and creating a list of dictionaries
data_list = []

for index, item in enumerate(data):
    data_dict = {}
    data_dict["id"] = index
    data_dict["question"] = item["question"]
    data_dict["imageURL"] = item["imageURL"]
    # itereate through the selected parts and objects and add them to the dictionary
    data_dict["selected_parts_polygon"] = []
    data_dict["selected_parts_text"] = []
    data_dict["selected_objects_polygon"] = []
    data_dict["selected_objects_text"] = []
    for part_index in item["selected_parts_polygons"]:
        data_dict["selected_parts_polygon"].append(
            item["parts_polygons"]["polygons"][part_index]
        )
        data_dict["selected_parts_text"].append(
            item["parts_labels"][part_index])

    for object_index in item["selected_objects_polygons"]:
        data_dict["selected_objects_polygon"].append(
            item["objects_polygons"][object_index])
        data_dict["selected_objects_text"].append(
            item["objects_labels"][object_index])
    data_list.append(data_dict)

# Save the processed data to a JSON file
output_file = "ambiguous_data.json"

with open(output_file, "w") as f:
    json.dump(data_list, f, indent=4)

# Notify user of successful export
output_file
