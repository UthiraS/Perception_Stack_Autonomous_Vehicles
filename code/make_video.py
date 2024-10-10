from moviepy.editor import ImageSequenceClip
import os

# Set the folder path and image file prefix
image_folder_path = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/Blender"
image_prefix = ""

# Set the output video path and filename
output_video_folder = "/Users/irakligrigolia/Desktop/WPI 22-23/CV/Einstein Vision/videos"
output_video_filename = "output_video.mp4"
output_video_path = os.path.join(output_video_folder, output_video_filename)

# Set the video resolution and framerate
resolution = (1920, 1080)
fps = 24

# Get the list of image files sorted by name
image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith(".png")])

# Create the full file paths for each image
image_paths = [os.path.join(image_folder_path, image) for image in image_files]

# Create an ImageSequenceClip using the image paths
clip = ImageSequenceClip(image_paths, fps=fps)

# Resize the clip to the desired resolution
resized_clip = clip.resize(resolution)

# Write the output video file
resized_clip.write_videofile(output_video_path)
