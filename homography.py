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
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count

# Define the destination points for the warped frame (same as original frame size)
dst_points = np.float32([[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]])

# Compute the homography matrix using manually selected points
h_matrix, _ = cv2.findHomography(ref_points, dst_points)

# Preview the warped result using the first frame
ret, first_frame = video_capture.read()
if ret:
    # Warp the first frame to preview the transformation
    preview_warped_frame = cv2.warpPerspective(first_frame, h_matrix, (frame_width, frame_height))

    # Resize the preview frame to 1920x1080 for display
    preview_resized = cv2.resize(preview_warped_frame, (1920, 1080))

    # Display the resized warped frame preview
    cv2.imshow('Warped Preview', preview_resized)
    print("Press any key to continue or 'q' to exit.")
    
    # Wait for user to press any key to continue with processing or 'q' to exit
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print("Processing cancelled by user.")
        video_capture.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Warped Preview")

# Reset the video capture to start processing from the first frame
video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Define codec and create a VideoWriter object to save the output video
output_filename = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
output_video = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Initialize a frame counter
frame_count = 0

# Main loop to process video frames and show progress
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break  # Exit when video finishes

    # Warp the original frame using the manually calculated homography matrix
    warped_frame = cv2.warpPerspective(frame, h_matrix, (frame_width, frame_height))

    # Write the warped frame to the output video
    output_video.write(warped_frame)

    # Update and print progress
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    print(f"Processing: {progress:.2f}% complete", end='\r')

    # Check for manual stop ('q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nProcessing stopped by user.")
        break

# Release video capture, writer, and close any remaining windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()

print(f"\nProcessed video saved as {output_filename}")
