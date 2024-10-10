import cv2

# Path to the input video file
input_file = '/home/uthira/Documents/GitHub/P3Data/P3Data/Sequences/scene10/Undist/2023-03-06_19-48-30-front_undistort.mp4'

# Path to the output directory where the images will be saved
output_dir = '/home/uthira/Documents/GitHub/P3Data/P3Data/Sequence_Images/scene10/'

# Open the video file for reading
video_capture = cv2.VideoCapture(input_file)

# Initialize a frame counter
frame_count = 0

# Loop through the video frames
while True:
    # Read the next frame from the video file
    ret, frame = video_capture.read()

    # If we've reached the end of the video file, break out of the loop
    if not ret:
        break

    # Increment the frame counter
    frame_count += 1

    # Construct the filename for this frame
    output_filename = f'{output_dir}/frame_{frame_count:04d}.jpg'

    # Save the frame as an image file
    cv2.imwrite(output_filename, frame)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()