import json
import random

# Load the data
ambigous_data = json.load(open('LIVE_0_2537_ambiguous_results.json'))
unambigous_data = json.load(open('LIVE_0_2537_unambiguous_results.json'))

# Inspect headers
print("Ambiguous data keys:", ambigous_data["4"].keys())
# print("Unambiguous data keys:", unambigous_data["4"].keys())

# Sort and match keys
matched_data = []

for key in ambigous_data.keys():
    if key in unambigous_data:
        # Match found, combine or compare
        # formatting ambigous_data[key] and unambigous_data[key] as needed

        # Step 1: check if the selected question is within the range of questions
        # if ambigous_data[key]["selected_question"] >= len(ambigous_data[key]["questions"]):
        #     for i in range(len(ambigous_data[key]["questions"]), ambigous_data[key]["selected_question"] + 1):
        #         ambigous_data[key]["questions"].append(ambigous_data[key]["questions"][-1])
        #     ambigous_data[key]["questions"][ambigous_data[key]["selected_question"]] = ambigous_data[key]["final_question"]

        # if unambigous_data[key]["selected_question"] >= len(unambigous_data[key]["questions"]):
        #     for i in range(len(unambigous_data[key]["questions"]), unambigous_data[key]["selected_question"] + 1):
        #         unambigous_data[key]["questions"].append(unambigous_data[key]["questions"][-1])
        #     unambigous_data[key]["questions"][unambigous_data[key]["selected_question"]] = unambigous_data[key]["final_question"]

        # Step 1: remove worker_id, original_questions, time_usage, selected_quesiton from the data
        ambigous_data[key].pop("worker_id")
        ambigous_data[key].pop("original_questions")
        ambigous_data[key].pop("time_usage")
        ambigous_data[key].pop("selected_question")

        ambigous_data[key]["question"] = ambigous_data[key]["final_question"]
        ambigous_data[key].pop("final_question")
        ambigous_data[key].pop("questions")

        # add ambigous as a key in ambigous_data[key]
        ambigous_data[key]["dataset"] = "PACO"

        # now check for selected parts and selected objects in ambigous_data[key]
        # if selected parts is not empty then add parts and masks to the data, else add objects and masks to the data
        selected_parts = ambigous_data[key]["selected_parts_polygons"]
        selected_objects = ambigous_data[key]["selected_objects_polygons"]

        # if all are empty or all are not empty then skip
        if len(selected_parts) == 0 and len(selected_objects) == 0:
            print("Both selected_parts and selected_objects are empty for key:", key)
            continue

        if len(selected_parts) > 0 and len(selected_objects) > 0:
            print("Both selected_parts and selected_objects are not empty for key:", key)
            continue

        ambigous_data[key]["polygons"] = []
        ambigous_data[key]["masks"] = []
        ambigous_data[key]["labels"] = []

        if len(selected_parts) != 0:

            for index in selected_parts:
                ambigous_data[key]["polygons"].append(ambigous_data[key]["parts_polygons"]["polygons"][index])
                ambigous_data[key]["masks"].append(ambigous_data[key]["parts_masks"][index])
                ambigous_data[key]["labels"].append(ambigous_data[key]["parts_labels"][index])
            ambigous_data[key]["type"] = "part"

        else:
            for index in selected_objects:
                ambigous_data[key]["polygons"].append(ambigous_data[key]["objects_polygons"][index])
                ambigous_data[key]["masks"] = []
                ambigous_data[key]["labels"].append(ambigous_data[key]["objects_labels"][index])
            ambigous_data[key]["type"] = "object"

        ambigous_data[key].pop("selected_parts_polygons")
        ambigous_data[key].pop("selected_objects_polygons")
        ambigous_data[key].pop("parts_polygons")
        ambigous_data[key].pop("parts_masks")
        ambigous_data[key].pop("parts_labels")
        ambigous_data[key].pop("objects_polygons")
        ambigous_data[key].pop("objects_labels")
        matched_data.append(ambigous_data[key])

        unambigous_data[key].pop("worker_id")
        unambigous_data[key].pop("original_questions")
        unambigous_data[key].pop("time_usage")
        unambigous_data[key].pop("selected_question")

        unambigous_data[key]["question"] = unambigous_data[key]["final_question"]
        unambigous_data[key].pop("final_question")
        unambigous_data[key].pop("questions")

        unambigous_data[key]["dataset"] = "PACO"

        # now check for selected parts and selected objects in ambigous_data[key]
        # if selected parts is not empty then add parts and masks to the data, else add objects and masks to the data
        selected_parts = unambigous_data[key]["selected_parts_polygons"]
        selected_objects = unambigous_data[key]["selected_objects_polygons"]

        # if all are empty or all are not empty then skip
        if len(selected_parts) == 0 and len(selected_objects) == 0:
            print("Both selected_parts and selected_objects are empty for key:", key)
            continue

        if len(selected_parts) > 0 and len(selected_objects) > 0:
            print(
                "Both selected_parts and selected_objects are not empty for key:", key
            )
            continue

        unambigous_data[key]["polygons"] = []
        unambigous_data[key]["masks"] = []
        unambigous_data[key]["labels"] = []
        if len(selected_parts) != 0:
            for index in selected_parts:
                unambigous_data[key]["polygons"].append(unambigous_data[key][
                    "parts_polygons"
                ]["polygons"][index])
                unambigous_data[key]["masks"].append(unambigous_data[key]["parts_masks"][
                    index
                ])
                unambigous_data[key]["labels"].append(unambigous_data[key]["parts_labels"][
                    index
                ])
            unambigous_data[key]["type"] = "part"
            

        else:
            for index in selected_objects:
                unambigous_data[key]["polygons"].append(unambigous_data[key][
                    "objects_polygons"
                ][index])
                unambigous_data[key]["masks"] = []
                unambigous_data[key]["labels"].append(unambigous_data[key]["objects_labels"][
                    index
                ])
            unambigous_data[key]["type"] = "object"
        unambigous_data[key].pop("selected_parts_polygons")
        unambigous_data[key].pop("selected_objects_polygons")
        unambigous_data[key].pop("parts_polygons")
        unambigous_data[key].pop("parts_masks")
        unambigous_data[key].pop("parts_labels")
        unambigous_data[key].pop("objects_polygons")
        unambigous_data[key].pop("objects_labels")
        matched_data.append(unambigous_data[key])

# Save the matched data
with open("paco.json", "w") as f:
    json.dump(matched_data, f, indent=4)

# save also a small subset for visualization
subset = random.sample(matched_data, 10)
with open("paco_subset.json", "w") as f:
    json.dump(subset, f, indent=4)
