import cv2

# Load the image
image = cv2.imread('/home/uthira/Documents/GitHub/MiDaS/Result/image_rgb.jpg')

# Convert the image to a NumPy array
image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the pixel value at (x, y) coordinates
x = (826.1404+856.4579)/2
y = (468.15497+543.44086)/2

pixel_value = image_array[int(y), int(x)]

print(pixel_value)