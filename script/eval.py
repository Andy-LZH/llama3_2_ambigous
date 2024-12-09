import re
import json
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry

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



def extract_points(zs_output):
    """
    Extract (x, y) points from a zs_output string.
    """
    pattern = r'x\d+="([0-9.]+)" y\d+="([0-9.]+)"'
    matches = re.findall(pattern, zs_output)
    return [(float(x), float(y)) for x, y in matches]


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


with open("data/molmo_output_non_processed.json", "r") as file:
    dataset = json.load(file)

for index, data in tqdm(enumerate(dataset)):
    zs_original_output = data["zs_original_output"]
    zs_molmo_output = data["zs_molmo_output"]

    zs_original_points = extract_points(zs_original_output)
    zs_molmo_points = extract_points(zs_molmo_output)

    # download and set image from cocodataset
    image_url = data["imageURL"]
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.convert("RGB")
    # convert to numpy array
    image = np.array(image)
    predictor.set_image(image)

    for index, points in enumerate(zs_original_points):
        masks = extract_masks(list(points))
        if masks is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_points(np.array([list(points)]), np.array([1]), plt.gca())
            show_mask(masks[0], plt.gca())
            plt.axis("on")
    