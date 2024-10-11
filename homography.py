import cv2
import numpy as np

# Global variables to store the points
ref_points = []
selected_points = 0
resize_factor = 0.5  # Adjust to downscale the 3840x2160 video to 1920x1080 for selection

# Mouse callback function to capture points on reference image
def select_point(event, x, y, flags, param):
    global ref_points, selected_points, resize_factor
    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust point coordinates to original 4K resolution
        original_x = int(x / resize_factor)
        original_y = int(y / resize_factor)
        ref_points.append([original_x, original_y])
        selected_points += 1
        # Draw a small circle on the selected point
        cv2.circle(resized_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Reference Image", resized_img)

# Load the reference image
reference_img = cv2.imread('Homography Reference.jpg')

# Resize the reference image for easier selection (1920x1080)
resized_img = cv2.resize(reference_img, (1920, 1080))
cv2.imshow("Reference Image", resized_img)
cv2.setMouseCallback("Reference Image", select_point)

# Wait until 4 points are selected
while selected_points < 4:
    if cv2.getWindowProperty('Reference Image', cv2.WND_PROP_VISIBLE) < 1:
        break
    cv2.waitKey(1)

cv2.destroyWindow("Reference Image")

# Convert points to a numpy array
ref_points = np.float32(ref_points)

# Load the 4K video (footage.mp4)
video_capture = cv2.VideoCapture('footage.mp4')

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

# Define the destination points for the warped frame
# These points represent the corners of the desired output (straightened) frame
dst_points = np.float32([[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]])

# Define codec and create a VideoWriter object to save the output video
output_filename = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
output_video = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Compute the homography matrix using manually selected points
h_matrix, _ = cv2.findHomography(ref_points, dst_points)

# Main loop to process video frames
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break  # Exit when video finishes

    # Warp the video frame using the manually calculated homography matrix
    warped_frame = cv2.warpPerspective(frame, h_matrix, (frame_width, frame_height))

    # Write the warped frame to the output video
    output_video.write(warped_frame)

    # Resize the warped frame for display to 1920x1080
    display_frame = cv2.resize(warped_frame, (1920, 1080))

    # Display the warped frame in a 1920x1080 window
    cv2.imshow('Warped Video', display_frame)

    # Press 'q' to exit the preview window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Or check if the window is closed manually
    if cv2.getWindowProperty('Warped Video', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release video capture, writer, and close windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_filename}")
