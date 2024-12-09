import json
import requests
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
    # Load dataset
    with open("data/ambiguous_data.json", "r") as file:
        dataset = json.load(file)

    # Load prompts
    prompts_grd = load_prompts("data/prompt/prompts_grd.json")
    prompts_grd_molmo = load_prompts("data/prompt/prompts_grd_molmo.json")

    # Extract original and molmo prompts
    zs_original_prompt = prompts_grd["ZS"]["prompt"]
    fs_original_prompt = prompts_grd["FS"]["prompt"]
    zs_molmo_prompt = prompts_grd_molmo["ZS"]["prompt"]
    fs_molmo_prompt = prompts_grd_molmo["FS"]["prompt"]

    # Initialize the model and processor
    model_name = "allenai/Molmo-7B-D-0924"
    processor, model = initialize_model_and_processor(model_name)

    molmo_output = []

    # Process the dataset
    for idx, item in enumerate(tqdm(dataset, desc="Processing items", unit="item")):
        question = item["question"]
        image_url = item["imageURL"]
        polygons = (
            item["selected_parts_polygon"]
            if item["selected_parts_polygon"]
            else item["selected_objects_polygon"]
        )
        labels = (
            item["selected_parts_text"]
            if item["selected_parts_polygon"]
            else item["selected_objects_text"]
        )

        # Generate responses for each prompt type
        zs_original_output = get_response(question, zs_original_prompt, image_url)
        fs_original_output = get_response(question, fs_original_prompt, image_url)
        zs_molmo_output = get_response(question, zs_molmo_prompt, image_url)
        fs_molmo_output = get_response(question, fs_molmo_prompt, image_url)

        # Append results
        molmo_output.append(
            {
                "id": item["id"],
                "question": question,
                "imageURL": image_url,
                "polygons": polygons,
                "labels": labels,
                "zs_original_output": zs_original_output,
                "fs_original_output": fs_original_output,
                "zs_molmo_output": zs_molmo_output,
                "fs_molmo_output": fs_molmo_output,
            }
        )
    # Save the output to a JSON file
    with open("data/molmo_output_non_processed.json", "w") as f:
        json.dump(molmo_output, f)
