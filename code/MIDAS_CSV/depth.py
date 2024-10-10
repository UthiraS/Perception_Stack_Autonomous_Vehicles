import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

filename = "/home/uthira/Documents/GitHub/MiDaS/test_input.jpg"
image = cv2.imread(filename)
h, w = image.shape[:2]

print(h,w)

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
    
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

h, w = output.shape[:2]

print(h,w)
# patch_size = 5  # size of the patch around the pixel

# # Extract a patch of pixels around the given coordinates
# patch = output[max(0, y-patch_size):min(y+patch_size, output.shape[0]),
#                max(0, x-patch_size):min(x+patch_size, output.shape[1])]

# # Compute the mean depth value for the patch
# avg_depth = np.mean(patch)

# print(avg_depth)
# print(h,w)
# # cv2.imwrite('/home/uthira/Documents/GitHub/MiDaS/Result/image_rgb.jpg', output)

# # gray_output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

# # Car coordinates with z:
# #             x           y   z
# x= 535.749420  
# y = 645.973055  

# # 16
# # x= 798.884375  
# # y =887.166150  
# # # 34
# # # Person coordinates with z:
# #         #   x          y   z
# # x =702.5834  
# # y =809.97465  
# # # 29
# # # traffic coordinates with z:
# #     #    x      y  z
# # x =521.0  
# # # y =566.5  8
# # # stop coordinates with z:
# #     #    x      y   z
# # x =609.0  
# # y =715.0  

# avg_depth = output(int(y),int(x))

# print("average depth :",avg_depth)

# # Convert the color image to grayscale
# # img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# # Save the grayscale image
# # cv2.imwrite('/home/uthira/Documents/GitHub/MiDaS/Result/image_gray.jpg', img_gray)

plt.imshow(output)
plt.show()
