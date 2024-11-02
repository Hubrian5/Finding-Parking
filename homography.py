import cv2
import numpy as np

# How to Use
# 1. Select 4 points on the reference image as a reference then press q
# 2. Select the 4 corresponding points on the video then press q
# 3. Wait until its done processing and its done


# Load the reference image
reference_image = cv2.imread('reference.jpg')
reference_points = []
display_scale = 0.5  # Adjust this value to fit your screen size

# Function to capture four points on the reference image
def select_reference_points(event, x, y, flags, param):
    global reference_points
    if event == cv2.EVENT_LBUTTONDOWN and len(reference_points) < 4:
        reference_points.append((int(x / display_scale), int(y / display_scale)))  # Scale back to original
        cv2.circle(reference_image_resized, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Reference Image", reference_image_resized)

# Resize and display the reference image to select points
reference_image_resized = cv2.resize(reference_image, None, fx=display_scale, fy=display_scale)
cv2.imshow("Reference Image", reference_image_resized)
cv2.setMouseCallback("Reference Image", select_reference_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(reference_points) != 4:
    print("Please select exactly four points on the reference image.")
    exit(1)

# Open the video file
video = cv2.VideoCapture('footage.mp4')

# Get total number of frames for progress tracking
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Read the first frame to select points
ret, first_frame = video.read()
if not ret:
    print("Error reading the video file.")
    exit(1)

video_points = []

# Function to capture four points on the video frame
def select_video_points(event, x, y, flags, param):
    global video_points
    if event == cv2.EVENT_LBUTTONDOWN and len(video_points) < 4:
        video_points.append((int(x / display_scale), int(y / display_scale)))  # Scale back to original
        cv2.circle(first_frame_resized, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("First Frame", first_frame_resized)

# Resize and display the first frame to select points
first_frame_resized = cv2.resize(first_frame, None, fx=display_scale, fy=display_scale)
cv2.imshow("First Frame", first_frame_resized)
cv2.setMouseCallback("First Frame", select_video_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(video_points) != 4:
    print("Please select exactly four points on the video frame.")
    exit(1)

# Compute the homography matrix
reference_pts = np.array(reference_points, dtype="float32")
video_pts = np.array(video_points, dtype="float32")
H, _ = cv2.findHomography(video_pts, reference_pts, method=cv2.RANSAC)

# Define the size of the output video to match the reference image
output_size = (reference_image.shape[1], reference_image.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('warped_video.mp4', fourcc, 30.0, output_size)

# Warp each frame and track progress
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Apply the homography transformation to warp the frame
    warped_frame = cv2.warpPerspective(frame, H, output_size)
    
    # Write the warped frame to the output video
    out.write(warped_frame)
    
    # Update progress
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    print(f"Processing frame {frame_count}/{total_frames} - {progress:.2f}% complete", end='\r')

# Release resources
video.release()
out.release()
cv2.destroyAllWindows()
print("\nProcessing complete.")
