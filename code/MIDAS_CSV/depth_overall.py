import cv2
import torch
import os
import matplotlib.pyplot as plt

# Load MiDaS model
model_type = "DPT_Large"  # MiDaS v3 - Large (highest accuracy, slowest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Input and output folders
input_folder = "/home/uthira/Documents/GitHub/P3Data/P3Data/Sequence_Images/scene1"
output_folder = "/home/uthira/Documents/GitHub/MiDaS/Overall/Depth_Images"

# Iterate over all the images in the input folder
i = 0
for filename in os.listdir(input_folder):

    print("filename :",filename)
    i = i +1
    if(i == 3):
        break
    # Load the image
    image = cv2.imread(os.path.join(input_folder, filename))
    h, w = image.shape[:2]

    # Apply MiDaS model to compute the depth map
    input_batch = transform(image).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()

    # Save the output depth map in the output folder
    output_filename = "depth_" + filename
    # cv2.imwrite(os.path.join(output_folder, output_filename), output)
    fig = plt.figure()
    plt.imshow(output)
    plt.show()
    plt.savefig(os.path.join(output_folder, output_filename),format='png', dpi=300)
