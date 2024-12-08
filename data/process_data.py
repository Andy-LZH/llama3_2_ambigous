import json
import random

# Load the data
ambigous_data = json.load(open('LIVE_0_2537_ambiguous_results.json'))
unambigous_data = json.load(open('LIVE_0_2537_unambiguous_results.json'))

# Inspect headers
print("Ambiguous data keys:", ambigous_data["4"].keys())
# print("Unambiguous data keys:", unambigous_data["4"].keys())

# Sort and match keys
matched_data = {
    "ambigous": [],
    "unambigous": []
}

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

        unambigous_data[key].pop("worker_id")
        unambigous_data[key].pop("original_questions")
        unambigous_data[key].pop("time_usage")
        unambigous_data[key].pop("selected_question")

        unambigous_data[key]["question"] = unambigous_data[key]["final_question"]
        unambigous_data[key].pop("final_question")
        unambigous_data[key].pop("questions")

        # Step 2: append the data to the matched_data
        matched_data["ambigous"].append(ambigous_data[key])
        matched_data["unambigous"].append(unambigous_data[key])

# do a random 80/20 split
data_length = len(matched_data["ambigous"])
indices = list(range(data_length))
random.shuffle(indices)

split_index = int(data_length * 0.8)
train_indices = indices[:split_index]
test_indices = indices[split_index:]

print("Train data length:", len(train_indices))
print("Test data length:", len(test_indices))

# Save the data

train_data = {
    "ambigous": [matched_data["ambigous"][i] for i in train_indices],
    "unambigous": [matched_data["unambigous"][i] for i in train_indices]
}

test_data = {
    "ambigous": [matched_data["ambigous"][i] for i in test_indices],
    "unambigous": [matched_data["unambigous"][i] for i in test_indices]
}

# print keys to ensure correctness
# print(train_data["ambigous"][0].keys())
# print(test_data["ambigous"][0].keys())

# print(train_data["unambigous"][0].keys())
# print(test_data["unambigous"][0].keys())

with open('train.json', 'w') as f:
    json.dump(train_data, f)

with open('test.json', 'w') as f:
    json.dump(test_data, f)