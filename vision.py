import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    if im1 is None or im2 is None or im1.size == 0 or im2.size == 0:
        return 0
    return np.abs(np.mean(im1) - np.mean(im2))


def validate_and_fix_spot(spot, frame_width, frame_height):
    x1, y1, w, h = spot

    # Ensure coordinates are within frame bounds
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))

    # Adjust width and height to fit within frame
    w = min(w, frame_width - x1)
    h = min(h, frame_height - y1)

    return [x1, y1, w, h]


# Load video first to get dimensions
video_path = 'cropped.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video frame dimensions: {frame_width}x{frame_height}")

# Load and resize mask to match video dimensions
mask = cv2.imread('mask.png', 0)
if mask is None:
    print("Error: Could not read mask file")
    exit()

print(f"Original mask dimensions: {mask.shape}")
mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
print(f"Resized mask dimensions: {mask.shape}")

# Threshold the resized mask to ensure binary values
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Get connected components
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# Get and validate parking spots
spots = get_parking_spots_bboxes(connected_components)
if not spots:
    print("Error: No parking spots found in mask")
    exit()

# Validate and adjust all spots to fit within frame
spots = [validate_and_fix_spot(spot, frame_width, frame_height) for spot in spots]

# Filter out invalid spots (too small)
MIN_WIDTH = 20
MIN_HEIGHT = 20
spots = [spot for spot in spots if spot[2] >= MIN_WIDTH and spot[3] >= MIN_HEIGHT]

print(f"Found {len(spots)} valid parking spots")
for i, spot in enumerate(spots):
    print(f"Spot {i}: {spot}")

spots_status = [None for _ in spots]
diffs = [None for _ in spots]

previous_frame = None
frame_nmr = 0
ret = True
step = 30

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            try:
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                prev_spot_crop = previous_frame[y1:y1 + h, x1:x1 + w, :]

                if spot_crop.size > 0 and prev_spot_crop.size > 0:
                    diffs[spot_indx] = calc_diff(spot_crop, prev_spot_crop)
            except Exception as e:
                print(f"Error processing spot {spot_indx}: {str(e)}")
                diffs[spot_indx] = 0

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            valid_diffs = [d for d in diffs if d is not None and d > 0]
            if valid_diffs:
                max_diff = max(valid_diffs)
                arr_ = [j for j in range(len(spots)) if diffs[j] is not None and diffs[j] / max_diff > 0.4]
            else:
                arr_ = range(len(spots))

        for spot_indx in arr_:
            try:
                x1, y1, w, h = spots[spot_indx]
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                if spot_crop.size > 0:
                    spot_status = empty_or_not(spot_crop)
                    spots_status[spot_indx] = spot_status
            except Exception as e:
                print(f"Error classifying spot {spot_indx}: {str(e)}")
                continue

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # Drawing
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        if spot_status is not None:
            x1, y1, w, h = spots[spot_indx]
            color = (0, 255, 0) if spot_status else (0, 0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    valid_spots = [s for s in spots_status if s is not None]
    if valid_spots:
        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(valid_spots)), str(len(valid_spots))),
                    (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()