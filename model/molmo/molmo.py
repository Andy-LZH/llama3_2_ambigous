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
    try:
        # Check whether the image is a link or a local file
        if image_url.startswith("http") or image_url.startswith("https"):
            image = Image.open(requests.get(image_url, stream=True).raw)
        else:
            image = Image.open(image_url)

        inputs = processor.process(
            images=[image],
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

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Argument parser to select from PACO, AnswerTherapy, or MSRA
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
        "MSRA": "data/MSRA/MSRA.json",
    }
    dataset_path = datasets[parser.parse_args().dataset]
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Load prompts
    prompts_grd = load_prompts("data/prompt/prompts_grd.json")

    # Extract original and molmo prompts
    ZS = prompts_grd["ZS"]["prompt"]

    # Initialize the model and processor
    model_name = "allenai/Molmo-7B-D-0924"
    processor, model = initialize_model_and_processor(model_name)

    molmo_output = []
    errors = []

    # Process the dataset
    for idx, item in enumerate(tqdm(dataset, desc="Processing items", unit="item")):
        try:
            question = item["question"]
            image_url = item["imageURL"]

            # Generate responses for each prompt type
            ZS_output = get_response(question, ZS, image_url)

            # Append results
            molmo_output.append(
                {
                    "id": idx,
                    "dataset": item["dataset"],
                    "ambiguity": item["ambiguity"],
                    "type": item.get("type", None),
                    "question": question,
                    "imageURL": image_url,
                    "polygons": item.get("polygons", None),
                    "masks": item.get("masks", None),
                    "labels": item.get("labels", None),
                    "ZS": ZS_output,
                    "focus_ambiguity_attribute": item.get(
                        "focus_ambiguity_attribute", None
                    ),
                }
            )
        except Exception as e:
            errors.append({"id": idx, "error": str(e), "item": item})

    # Save the output to a JSON file
    output_path = dataset_path.replace(".json", "_molmo.json")
    with open(output_path, "w") as f:
        json.dump(molmo_output, f)
    print(f"Results saved to {output_path}")

    # Save the errors to a separate JSON file
    error_path = dataset_path.replace(".json", "_errors.json")
    with open(error_path, "w") as f:
        json.dump(errors, f)
    print(f"Errors saved to {error_path}")
