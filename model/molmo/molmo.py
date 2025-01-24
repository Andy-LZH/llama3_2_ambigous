import json
import requests
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def load_prompts(file_path):
    """Load prompts from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def initialize_model_and_processor(model_name):
    """Initialize the model and processor."""
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    return processor, model


def get_response(question, prompt, image_url):
    """Generate a response using the model."""
    inputs = processor.process(
        images=[Image.open(requests.get(image_url, stream=True).raw)],
        text=prompt.format(question=question),
    )

    # Move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate output
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )

    # Decode the generated tokens to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    # argument parser to select from PACO, AnswerTherapy, or MSRA
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PACO", "AnswerTherapy", "MSRA"],
        required=True,
    )

    # Load dataset
    datasets = {
        "PACO": "data/paco/paco.json",
        "AnswerTherapy": "data/AnswerTherapy/answer_therapy.json",
        "MSRA": "data/MSRA/MSRA_RLE_627.json",
    }
    dataset_path = datasets[parser.parse_args().dataset]
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Load prompts
    prompts_grd = load_prompts("data/prompt/prompts_grd.json")

    # Extract original and molmo prompts
    ZS = prompts_grd["ZS"]["prompt"]
    ZS_CoT = prompts_grd["ZS_CoT"]["prompt"]
    ZS_ECoT = prompts_grd["ZS_ECoT"]["prompt"]
    FS = prompts_grd["FS"]["prompt"]
    FS_ECoT = prompts_grd["FS_ECoT"]["prompt"]

    # Initialize the model and processor
    model_name = "allenai/Molmo-7B-D-0924"
    processor, model = initialize_model_and_processor(model_name)

    molmo_output = []

    # Process the dataset
    for idx, item in enumerate(tqdm(dataset, desc="Processing items", unit="item")):
        question = item["question"]
        image_url = item["imageURL"]

        # Generate responses for each prompt type
        ZS_output = get_response(question, ZS, image_url)
        # ZS_CoT_output = get_response(question, ZS_CoT, image_url)
        # ZS_ECoT_output = get_response(question, ZS_ECoT, image_url)
        # FS_output = get_response(question, FS, image_url)
        # FS_ECoT_output = get_response(question, FS_ECoT, image_url)

        # Append results
        molmo_output.append(
            {
                "id": idx,
                "dataset": item["dataset"],
                "ambiguity": item["ambiguity"],
                "type": item["type"],
                "question": question,
                "imageURL": image_url,
                "polygons": item["polygons"],
                "masks": item["masks"],
                "labels": item["labels"],
                "ZS": ZS_output,
                # if AnswerTherapy also add focus_ambiguity_attribute
                "focus_ambiguity_attribute": item.get("focus_ambiguity_attribute", None),
            }
        )
        if idx == 20:
            break
    # Save the output to a JSON file
    # load directory of dataset path and save molmo_output
    output_path = dataset_path.replace(".json", "_molmo.json")
    with open(output_path, "w") as f:
        json.dump(molmo_output, f)
    print(f"Results saved to {output_path}")
