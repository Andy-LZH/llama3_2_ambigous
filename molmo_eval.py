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
from script.utils import loader
import wandb
import argparse
import os


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


def handling_data_point(data: dict):
    """
    Handling data point
    """
    image_url = data["imageURL"]
    if image_url.startswith("http") or image_url.startswith("https"):
        image = Image.open(requests.get(image_url, stream=True).raw)
    else:
        image = Image.open(image_url)

    image = image.convert("RGB")
    image = np.array(image)
    image_height, image_width, _ = image.shape

    molmo_output = data.get("ZS", None)
    return molmo_output, image, image_height, image_width


# define main function here
if __name__ == "__main__":

    # argparse different datasets
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PACO", "AnswerTherapy", "MSRA"],
        required=True,
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        required=False,
    )

    location_prefix_dict = {
        "PACO": "data/paco/",
        "AnswerTherapy": "data/AnswerTherapy/",
        "MSRA": "data/MSRA/",
    }

    # create gt_masks and pred_masks folders
    for dataset in location_prefix_dict.values():
        gt_masks_path = f"{dataset}/gt_masks"
        pred_masks_path = f"{dataset}/pred_masks"
        os.makedirs(gt_masks_path, exist_ok=True)
        os.makedirs(pred_masks_path, exist_ok=True)

    # Load dataset
    location_prefix = location_prefix_dict[parser.parse_args().dataset]
    dataset = loader(parser.parse_args().dataset, parser.parse_args().test)
    print("Dataset loaded.")

    wandb.init(
        project="Molmo ZS Evaluation {dataset}".format(
            dataset=parser.parse_args().dataset
        ),
        config={
            "model": "Molmo+SAM",
            "dataset": parser.parse_args().dataset,
            "checkpoint": "/scratch/alpine/zhli3162/.cache/sam/sam_vit_h_4b8939.pth",
            "molmo_model": "Molmo-7B-D-0924",
        },
    )

    sam = sam_model_registry["default"](
        checkpoint="/scratch/alpine/zhli3162/.cache/sam/sam_vit_h_4b8939.pth"
    )
    predictor = SamPredictor(sam)

    outputs = []

    union_iou_zs = 0
    for index_id, data in enumerate(
        tqdm(dataset, desc="Processing items", unit="item")
    ):
        # try:

        output_dict = {
            "id": data["id"],
            "question": data["question"],
            "imageURL": data["imageURL"],
            "ambiguity": data["ambiguity"],
            "type": data.get("type", None),
            "focus_ambiguity_attribute": data.get("focus_ambiguity_attribute", None),
            "zs_molmo_output": data.get("ZS", None),
            "gt_masks_location": None,
            "pred_masks_location": None,
            "mAP": None,
            "Union IoU": None,
            "Max IoU": None,
        }

        # Download and set image, and get image height and width, and molmo output
        molmo_output, image, image_height, image_width = handling_data_point(data)
        print(image.shape)
        predictor.set_image(image)
        molmo_points = extract_points(molmo_output, image_width, image_height)

        print("Processing item", index_id)

        if molmo_points is not None:
            masks_molmo = []
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for index, points in enumerate(molmo_points):
                masks = extract_masks(list(points))
                if masks is not None:
                    show_points(np.array([list(points)]), np.array([1]), plt.gca())
                    show_mask(masks[0], plt.gca())
                masks_molmo.append(masks)
            plt.savefig(f"{location_prefix}/pred_masks/id_{index_id}_molmo.png")
            plt.close()
            output_dict["pred_masks_location"] = (
                f"{location_prefix}/pred_masks/id_{index_id}_mask_x_molmo.png"
            )

        # Calculate the union IoU
        focus_regions = data["polygons"]

        binary_masks = nested_polygons_to_binary_mask(
            focus_regions, image_height, image_width
        ).astype(np.uint8)

        # draw the gt masks
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for index, mask in enumerate(binary_masks):
            show_mask(mask, plt.gca())
        plt.savefig(f"{location_prefix}/gt_masks/id_{index_id}_gt.png")
        plt.close()
        output_dict["gt_masks_location"] = (
            f"{location_prefix}/gt_masks/id_{index_id}_gt.png"
        )

        if molmo_points is not None:
            avg_iou_zs = 0
            max_iou_zs = 0
            for mask in masks_molmo:
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
                    max_iou_zs = max(max_iou_zs, iou)
                avg_iou_zs += max_iou_zs
            union_iou_zs += avg_iou_zs / len(mask)
            output_dict["Max IoU"] = avg_iou_zs / len(masks)

        wandb.log(output_dict)
        outputs.append(output_dict)

        # except Exception as e:
        #     wandb.log({"error": str(e), "step": index_id})
        #     print(e)
        #     print("Error processing item", index_id)
        #     continue

    wandb.log(
        {
            "final_iou": union_iou_zs / len(dataset),
            "num_samples": len(dataset),
        }
    )

    with open(
        f"{location_prefix}/molmo_zs_eval_{parser.parse_args().dataset}.json", "w"
    ) as f:
        json.dump(outputs, f)
