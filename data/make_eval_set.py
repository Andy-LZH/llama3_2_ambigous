import json
import os
import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Load JSON data
with open("ambiguous_data.json") as f:
    data = json.load(f)

# Prepare directories for saving images and masks
os.makedirs("images", exist_ok=True)
os.makedirs("masks_gt", exist_ok=True)

csv_data = []

# Load a default font for labels
font = ImageFont.load_default()

# Process each item in the data
for item in data:
    question = item["question"]
    image_url = item["imageURL"]
    if len(item["selected_parts_polygon"]) != 0:
        polygons = item["selected_parts_polygon"]
        labels = item["selected_parts_text"]
    else:
        polygons = item["selected_objects_polygon"]
        labels = item["selected_objects_text"]

    # Download and save the image
    image_name = f"{item['id']}_original.jpg"
    image_path = os.path.join("images", image_name)
    response = requests.get(image_url)
    with open(image_path, "wb") as img_file:
        img_file.write(response.content)

    # Open the original image
    original_image = Image.open(image_path)
    original_draw = ImageDraw.Draw(original_image)

    # Create a blank mask
    mask_image = Image.new("RGB", original_image.size, (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)

    # Draw polygons and overlay labels
    for polygon, label in zip(polygons, labels):
        for each_polygon in polygon:
            flat_polygon = [(x, y) for x, y in zip(each_polygon[::2], each_polygon[1::2])]

            # Draw polygon on mask
            mask_draw.polygon(flat_polygon, fill=(
                255, 255, 255), outline=(255, 255, 255))

            # Draw polygon on the original image
            original_draw.polygon(flat_polygon, outline="red")

            # Calculate the center of the polygon for label placement
            x_coords = [p[0] for p in flat_polygon]
            y_coords = [p[1] for p in flat_polygon]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)

            # Overlay the label on the mask
            mask_draw.text((center_x, center_y), label, fill="white", font=font)

        # Overlay the label on the original image
        mask_draw.text((center_x, center_y), label, fill="red", font=font)
    # Save the mask
    mask_name = f"{item['id']}_mask.png"
    mask_path = os.path.join("masks_gt", mask_name)
    mask_image.save(mask_path)

    # Save the labeled original image
    labeled_image_name = f"{item['id']}_labeled.jpg"
    labeled_image_path = os.path.join("images", labeled_image_name)
    original_image.save(labeled_image_path)

    molmo_image_path = os.path.join("molmo", f"{item['id']}_molmo.png")

    # Append to CSV data
    csv_data.append({
        "question": question,
        "original_image": image_path,
        "labeled_image": labeled_image_path,
        "mask_image": mask_path,
        "molmo_image": molmo_image_path
    })

# Save CSV file
csv_file_path = "output_data_with_labels.csv"
df = pd.DataFrame(csv_data)
df.to_csv(csv_file_path, index=False)

print("Processing complete. Images, masks, and CSV saved in the working directory.")
