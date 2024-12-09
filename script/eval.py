import re
import json
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
from pycocotools.mask import decode, encode, area
from matplotlib.path import Path
from pycocotools.cocoeval import COCOeval

import wandb

wandb.init(
    project="SAM_Score_Molmo",
    config={
        "model": "SAM",
        "dataset": "Focus Ambiguity VQA Dataset",
        "checkpoint": "/scratch/alpine/zhli3162/.cache/sam/sam_vit_h_4b8939.pth",
    },
)

sam = sam_model_registry["default"](
    checkpoint="/scratch/alpine/zhli3162/.cache/sam/sam_vit_h_4b8939.pth"
)
predictor = SamPredictor(sam)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def extract_points(zs_output, image_width, image_height):
    """
    Extract (x, y) points from a zs_output string.
    """
    pattern = r'x\d*=["\']?([0-9.]+)["\']?\s+y\d*=["\']?([0-9.]+)["\']?'
    matches = re.findall(pattern, zs_output)
    if not matches:
        return None
    return [
        (float(x) / 100 * image_width, float(y) / 100 * image_height)
        for x, y in matches
    ]


def extract_masks(input_points):
    """
    Extract masks from a zs_output string.
    """
    points = input_points
    if points:
        input_point = np.array([points])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=[1],
            box=None,
            multimask_output=False,
        )
        return masks
    return None


def polygon_to_binary_mask(polygon, image_height, image_width):
    """
    Convert a single polygon to a binary mask.

    Args:
        polygon: List of points defining the mask boundary (e.g., [x1, y1, x2, y2, ...]).
        image_height: Height of the image.
        image_width: Width of the image.

    Returns:
        binary_mask: 2D NumPy array representing the binary mask.
    """
    # Create a blank binary mask
    binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Create a Path object from the polygon
    poly_path = Path(np.array(polygon).reshape(-1, 2))

    # Create a grid of coordinates corresponding to the image
    x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))
    coords = np.stack((x.flatten(), y.flatten()), axis=-1)

    # Determine which points fall inside the polygon
    inside = poly_path.contains_points(coords)

    # Reshape the boolean array into the shape of the image
    binary_mask[inside.reshape((image_height, image_width))] = 1

    return binary_mask


def nested_polygons_to_binary_mask(nested_polygons, image_height, image_width):
    """
    Convert nested polygons to a combined binary mask.

    Args:
        nested_polygons: Nested list of polygons, where each polygon is a list of coordinates.
        image_height: Height of the image.
        image_width: Width of the image.

    Returns:
        combined_mask: 2D NumPy array representing the combined binary mask.
    """
    combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Iterate over all top-level polygon groups
    masks = []
    for polygons in nested_polygons:
        # Each `polygons` can itself be a list of polygons
        for polygon in polygons:
            mask = polygon_to_binary_mask(polygon, image_height, image_width)
            masks.append(mask)
    return np.array(masks)


with open("data/molmo_output_non_processed.json", "r") as file:
    dataset = json.load(file)

outputs = []

union_iou_zs_original = 0
union_iou_zs_molmo = 0
for index_id, data in enumerate(tqdm(dataset, desc="Processing items", unit="item")):
    try:

        output_dict = {
            "id": data["id"],
            "question": data["question"],
            "imageURL": data["imageURL"],
            "zs_original_output": None,
            "zs_molmo_output": None,
        }
        # Download and set image
        image_url = data["imageURL"]
        image = Image.open(requests.get(image_url, stream=True).raw)
        image = image.convert("RGB")
        # Convert to numpy array
        image = np.array(image)
        predictor.set_image(image)
        image_height, image_width, _ = image.shape

        zs_original_output = data["zs_original_output"]
        zs_molmo_output = data["zs_molmo_output"]

        zs_original_points = extract_points(
            zs_original_output, image_width, image_height
        )
        zs_molmo_points = extract_points(zs_molmo_output, image_width, image_height)

        print("Processing item", index_id)
        print(zs_original_points)
        print(zs_molmo_points)

        if zs_original_points is not None:
            masks_original = []
            for index, points in enumerate(zs_original_points):
                masks = extract_masks(list(points))
                if masks is not None:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    show_points(np.array([list(points)]), np.array([1]), plt.gca())
                    show_mask(masks[0], plt.gca())
                    plt.savefig(
                        f"data/masks_pred/id_{index_id}_mask_{index}_original.png"
                    )
                masks_original.append(masks)

            output_dict["zs_original_output"] = masks_original

        if zs_molmo_points is not None:
            masks_molmo = []
            for index, points in enumerate(zs_molmo_points):
                masks = extract_masks(list(points))
                if masks is not None:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    show_points(np.array([list(points)]), np.array([1]), plt.gca())
                    show_mask(masks[0], plt.gca())
                    plt.savefig(f"data/masks_pred/id_{index_id}_mask_{index}_molmo.png")
                masks_molmo.append(masks)

        # Calculate the union IoU
        focus_regions = data["polygons"]

        binary_masks = nested_polygons_to_binary_mask(
            focus_regions, image_height, image_width
        ).astype(np.uint8)

        print("Binary mask shape:", binary_masks.shape)

        if zs_original_points is not None:
            avg_iou_zs_original = 0
            for mask in masks_original:
                max_iou_zs_original = 0
                mask = mask[0]  # Extract the first mask from the prediction output
                mask = mask.astype(np.uint8)
                mask = mask.reshape(image_height, image_width)

                # Loop through all ground truth masks
                for binary_mask in binary_masks:
                    # Compute IoU
                    intersection = np.logical_and(mask, binary_mask)
                    union = np.logical_or(mask, binary_mask)
                    iou = np.sum(intersection) / np.sum(union)

                    # Update maximum IoU if the current one is higher
                    max_iou_zs_original = max(max_iou_zs_original, iou)
                avg_iou_zs_original += max_iou_zs_original
            union_iou_zs_original += avg_iou_zs_original / len(masks_original)
            output_dict["zs_original_output"] = avg_iou_zs_original / len(masks_original)
        
        if zs_molmo_points is not None:
            avg_iou_zs_molmo = 0
            for mask in masks_molmo:
                max_iou_zs_molmo = 0
                mask = mask[0]
                mask = mask.astype(np.uint8)
                mask = mask.reshape(image_height, image_width)

                for binary_mask in binary_masks:
                    intersection = np.logical_and(mask, binary_mask)
                    union = np.logical_or(mask, binary_mask)
                    iou = np.sum(intersection) / np.sum(union)
                    max_iou_zs_molmo = max(max_iou_zs_molmo, iou)
                avg_iou_zs_molmo += max_iou_zs_molmo
            union_iou_zs_molmo += avg_iou_zs_molmo / len(masks_molmo)
            output_dict["zs_molmo_output"] = avg_iou_zs_molmo / len(masks_molmo)

        wandb.log(output_dict)
        outputs.append(output_dict)

    except Exception as e:
        wandb.log({"error": str(e), "step": index_id})
        print(e)
        print("Error processing item", index_id)
        continue

wandb.log(
    {
        "final_iou_original": union_iou_zs_original / len(dataset),
        "final_iou_molmo": union_iou_zs_molmo / len(dataset),
    }
)
with open("data/molmo_output_processed.json", "w") as file:
    json.dump(outputs, file)
print("Processing complete. Results logged to Weights & Biases.")
