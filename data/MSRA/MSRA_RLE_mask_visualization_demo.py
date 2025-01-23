import json
import numpy as np
import matplotlib.pyplot as plt
import base64

import cv2

import pycocotools._mask as _mask

from PIL import Image

with open("MSRA_RLE_627.json","r") as f:
    msra_RLE = json.load(f)

def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]
    
for file in msra_RLE:
    # Visualize the first segmentation mask
    mask_info = msra_RLE[file]["RLE"][0]

    # Decode the base64-encoded RLE counts
    image_path = "MSRA_Image/" + file
    image = np.array(Image.open(image_path))

    # Decode the RLE mask
    rle_encoded = mask_info["counts"]
    mask_info["counts"] = base64.b64decode(rle_encoded)
    binary_mask = decode(mask_info)

    # Ensure binary_mask values are correctly scaled
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Find contours of the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert image to BGR if it's grayscale (OpenCV default color space is BGR)
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:  # Check RGB and convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Overlay color in BGR format (since OpenCV uses BGR)
    overlay_color =(240, 176, 0)#(0, 176, 240)#(0, 192, 255) # BGR for #A9D18E
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):  # Apply the color to each channel
        mask_overlay[:, :, i] = (binary_mask / 255 * overlay_color[i]).astype(np.uint8)

    # Blend the mask overlay with the original image
    blended_image = cv2.addWeighted(image, 1, mask_overlay, 0.3, 0)

    # Draw the solid outline in the overlay color
    cv2.drawContours(blended_image, contours, -1, overlay_color, thickness=3)

    # Save the result
    output_path = "MSRA_Overlay/" + msra_RLE[file]["question"][:-1]+file
    cv2.imwrite(output_path, blended_image)
    print(msra_RLE[file]["question"])