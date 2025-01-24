import json
import random

img_location = "/scratch/alpine/zhli3162/train2014/"
# Load the data
ambigous_data = json.load(open("AnswerTherapy_ambiguous_grd.json"))
unambigous_data = json.load(open("AnswerTherapy_unambiguous_grd.json"))

matched_data = []

for key in range(len(ambigous_data)):
    ambigous_data[key]["imageURL"] = img_location + ambigous_data[key]["file_name"]
    ambigous_data[key]["ambiguity"] = "True"
    ambigous_data[key]["question"] = ambigous_data[key].pop("question")
    ambigous_data[key]["dataset"] = "AnswerTherapy"
    ambigous_data[key]["masks"] = []
    ambigous_data[key]["labels"] = []
    ambigous_data[key]["type"] = "none"
    ambigous_data[key]["focus_ambiguity_attributes"] = ambigous_data[key].pop("focus_ambiguity_attributes")
    ambigous_data[key]["polygons"] = ambigous_data[key]["focus_grounding"]

    ambigous_data[key].pop("answer_type")
    ambigous_data[key].pop("answers")
    ambigous_data[key].pop("binary_label")
    ambigous_data[key].pop("file_name")
    ambigous_data[key].pop("height")
    ambigous_data[key].pop("image_filename")
    ambigous_data[key].pop("image_id")
    if "question_id" in ambigous_data[key].keys():
        ambigous_data[key].pop("question_id")
    ambigous_data[key].pop("width")
    ambigous_data[key].pop("ambiguous_question")
    ambigous_data[key].pop("grounding_labels")
    ambigous_data[key].pop("focus_grounding")
    ambigous_data[key].pop("entity_lookup")
    matched_data.append(ambigous_data[key])

    unambigous_data[key]["imageURL"] = img_location + unambigous_data[key]["file_name"]
    unambigous_data[key]["ambiguity"] = "True"
    unambigous_data[key]["question"] = unambigous_data[key].pop("question")
    unambigous_data[key]["dataset"] = "AnswerTherapy"
    unambigous_data[key]["masks"] = []
    unambigous_data[key]["labels"] = []
    unambigous_data[key]["type"] = "none"
    unambigous_data[key]["focus_ambiguity_attributes"] = unambigous_data[key].pop("focus_ambiguity_attributes")
    unambigous_data[key]["polygons"] = unambigous_data[key]["focus_grounding"]

    unambigous_data[key].pop("answer_type")
    unambigous_data[key].pop("answers")
    unambigous_data[key].pop("binary_label")
    unambigous_data[key].pop("file_name")
    unambigous_data[key].pop("height")
    unambigous_data[key].pop("image_filename")
    unambigous_data[key].pop("image_id")

    if "question_id" in unambigous_data[key].keys():
        unambigous_data[key].pop("question_id")

    unambigous_data[key].pop("width")
    unambigous_data[key].pop("ambiguous_question")
    unambigous_data[key].pop("grounding_labels")
    unambigous_data[key].pop("focus_grounding")
    unambigous_data[key].pop("entity_lookup")
    matched_data.append(unambigous_data[key])

# Save the matched data
with open("answer_therapy.json", "w") as f:
    json.dump(matched_data, f, indent=4)

# save also a small subset for visualization
subset = random.sample(matched_data, 10)
with open("answer_therapy_subset.json", "w") as f:
    json.dump(subset, f, indent=4)
