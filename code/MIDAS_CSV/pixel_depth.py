import cv2
from PIL import Image
# Load the image
image = cv2.imread('/home/uthira/Documents/GitHub/MiDaS/Result/Figure_1.png')
resized = cv2.resize(image, (1280,960), interpolation = cv2.INTER_AREA)

# Convert the image to a NumPy array
image_array = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

h, w = image.shape[:2]

print(w,h)



# h1, w1 = image_array.shape[:2]

# print(w1,h1)

# Get the pixel value at (x, y) coordinates
# x = 480
# y = 1279

# xmin = 610.47314
# xmax = 726.31165
# ymin = 461.0257
# ymax = 565.63446
# xmin =1145.5735
# xmax =1278.4949
# ymin =452.19525
# ymax =495.8374
# xmin =932
# xmax =1042
# ymin =286
# ymax =388
# xmin =967
# xmax =1004
# ymin =75
# ymax =129
xmin = 994.5127
xmax =1043.9893
ymin =410.6541
ymax =575.96

x = xmin + (-xmin+xmax)/2
y = ymin + (-ymin+ymax)/2

# x = 640
# y = 959
# print("center_x",center_x)
# print("center_y",center_y)
pixel_value = image_array[int(y),int(x)]
print("pixel_value  at x,y:",pixel_value,x,y)
gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
print("grayscale :",gray_scale)

# pixel_value = image_array[int(center_x),int(center_y)]

# print(pixel_value)
# for x in range(0,1280,10):
#     for y in range(0,960,10):
#         # x = 640
#         # y = 959
#         pixel_value = image_array[int(y),int(x)]

#         print("pixel_value  at x,y:",pixel_value,x,y)
#         gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
#         print("grayscale :",gray_scale)


# img = Image.open("/home/uthira/Documents/GitHub/MiDaS/Result/image_rgb.jpg")



# # If the image is in RGB format, convert it to grayscale first
# if img.mode == "RGB":
#     img = img.convert("L")

# # Get grayscale value for pixel at x, y coordinates
# for x in range(0,1280,10):
#     for y in range(0,960,10):
#         # x = 640
#         # y = 959
#         pixel_value = img.getpixel((x, y))

#         print("pixel_value  at x,y:",pixel_value,x,y)

# xmin = 610.47314
# xmax = 726.31165
# ymin = 461.0257
# ymax = 565.63446

# center_x = (xmin+xmax)/2
# center_y = (ymin+ymax)/2

# pixel_value = img.getpixel((center_x, center_y))
# print("pixel_value  at x,y:",pixel_value,center_x,center_y)


# xmin =1145.5735
# xmax =1278.4949
# ymin =452.19525
# ymax =495.8374

# center_x = (xmin+xmax)/2
# center_y = (ymin+ymax)/2

# pixel_value = img.getpixel((center_x, center_y))
# print("pixel_value  at x,y:",pixel_value,center_x,center_y)
