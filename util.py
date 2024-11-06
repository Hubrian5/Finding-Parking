import pickle
from skimage.transform import resize
import numpy as np
import cv2

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.pkl", "rb"))


def empty_or_not(spot_bgr):
    # Add input validation
    if spot_bgr is None or spot_bgr.size == 0:
        print("Error: Empty input image")
        return EMPTY

    # Print debug information
    print(f"Input image shape: {spot_bgr.shape}")

    # Ensure the input image has valid dimensions
    if len(spot_bgr.shape) != 3:
        print("Error: Input image must be 3-dimensional (height, width, channels)")
        return EMPTY

    try:
        # Make sure we resize to exactly the same dimensions as training
        img_resized = cv2.resize(spot_bgr, (15, 20))  # Using cv2.resize instead of skimage.transform.resize
        img_resized = img_resized / 255.0  # Normalize to match training data

        # Ensure we have 3 channels
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        flat_data = []
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)

        print(f"Processed feature vector shape: {flat_data.shape}")

        y_output = MODEL.predict(flat_data)

        return EMPTY if y_output == 0 else NOT_EMPTY

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(f"Spot BGR shape: {spot_bgr.shape}")
        return EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):
        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        # Add validation for negative values or zero dimensions
        if x1 < 0 or y1 < 0 or w <= 0 or h <= 0:
            print(f"Warning: Invalid bounding box detected: x1={x1}, y1={y1}, w={w}, h={h}")
            continue

        slots.append([x1, y1, w, h])

    return slots