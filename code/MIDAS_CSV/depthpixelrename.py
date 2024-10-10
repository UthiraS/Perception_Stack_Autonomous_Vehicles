import numpy as np
import pandas as pd
import cv2
from csv import reader

def load_csv(filename):
    # Open file in read mode
    file = open(filename, "r")
    # Reading file
    lines = reader(file)
    
    # Converting into a list
    data = list(lines)
    return data

# Read car and person coordinates CSV files
car_coordinates_csv = 'bounding_boxes_2.csv'
person_coordinates_csv = 'bounding_boxes_0.csv'
traffic_coordinates_csv = '/home/uthira/Documents/GitHub/MiDaS/trafficlight.csv'
stop_coordinates_csv = '/home/uthira/Documents/GitHub/MiDaS/stop.csv'

line_coordinates_csv = '/home/uthira/Documents/GitHub/MiDaS/line_coordinates.csv'

car_coordinates_list = load_csv(car_coordinates_csv)[1:]  # skip header row
person_coordinates_list = load_csv(person_coordinates_csv)[1:]  # skip header row
traffic_coordinates_list = load_csv(traffic_coordinates_csv)[1:]  # skip header row
stop_coordinates_list = load_csv(stop_coordinates_csv)[1:]  # skip header row


line_coordinates_list = load_csv(line_coordinates_csv)[1:]  # skip header row


# Initialize empty lists for storing pixel coordinates and depth values
car_pixel_coordinates = []
person_pixel_coordinates = []
traffic_pixel_coordinates = []
stop_pixel_coordinates = []
line_pixel_coordinates = []

image = cv2.imread('/home/uthira/Documents/GitHub/MiDaS/Result/Figure_1.png')
resized = cv2.resize(image, (1280,960), interpolation = cv2.INTER_AREA)

# Convert the image to a NumPy array
image_array = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

h, w = image.shape[:2]

print(w,h)

# Iterate over each row in the car coordinates CSV file
for i, row in enumerate(car_coordinates_list):
    # Read the image file
    # image_path = '/home/uthira/Documents/GitHub/MiDaS/Result/image_rgb.jpg'  
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the x and y coordinates
    x1, x2, y1, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

   

    # Calculate the average depth value
    pixel_value = image_array[int(center_y),int(center_x)]
    print("pixel_value  at x,y:",pixel_value,center_x,center_y)
    gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
    print("grayscale :",gray_scale)
    # avg_depth = image[int(center_y), int(center_x)]

    # if avg_depth <-1:
    #     avg_depth = 1

    # Append the pixel coordinates and depth value to the stop pixel coordinates list
    car_pixel_coordinates.append([center_x, 255-gray_scale,0])

# Iterate over each row in the person coordinates CSV file
for i, row in enumerate(person_coordinates_list):
    # Read the image file
    # image_path = '/home/uthira/Documents/GitHub/MiDaS/Result/image_rgb.jpg'  
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the x and y coordinates
    x1, x2, y1, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    print(x1,y1,x2,y2)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the average depth value
    pixel_value = image_array[int(center_y),int(center_x)]
    print("pixel_value  at x,y:",pixel_value,center_x,center_y)
    gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
    print("grayscale :",gray_scale)
    # avg_depth = image[int(center_y), int(center_x)]

    # if avg_depth <-1:
    #     avg_depth = 1

    # Append the pixel coordinates and depth value to the stop pixel coordinates list
    person_pixel_coordinates.append([center_x, 255-gray_scale,0])


# Iterate over each row in the traffic coordinates CSV file
for i, row in enumerate(traffic_coordinates_list):
    # Read the image file
    # image_path = '/home/uthira/Documents/GitHub/MiDaS/Result/image_rgb.jpg'  
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(row[0])
    print(row[1])
    print(row[2])
    print(row[3])
    # Get the x and y coordinates
    x1, x2, y1, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    print(x1,y1,x2,y2)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

     # Calculate the average depth value
    pixel_value = image_array[int(center_y),int(center_x)]
    print("pixel_value  at x,y:",pixel_value,center_x,center_y)
    gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
    print("grayscale :",gray_scale)
    # avg_depth = image[int(center_y), int(center_x)]

    # if avg_depth <-1:
    #     avg_depth = 1

    # Append the pixel coordinates and depth value to the stop pixel coordinates list
    
    center_y = int(center_y)
    print("center_y ",center_y)
    if ((center_y >0) & (center_y<480)):
        traffic_pixel_coordinates.append([center_x, 255-gray_scale,center_y])
    else :
        traffic_pixel_coordinates.append([center_x, 255-gray_scale,0])
    


# Iterate over each row in the stop coordinates CSV file
for i, row in enumerate(stop_coordinates_list):
    # Read the image file
    # image_path = '/home/uthira/Documents/GitHub/MiDaS/Result/image_rgb.jpg'  
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(row[0])
    print(row[1])
    print(row[2])
    print(row[3])
    # Get the x and y coordinates
    x1, x2, y1, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    print(x1,y1,x2,y2)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the average depth value
    pixel_value = image_array[int(center_y),int(center_x)]
    print("pixel_value  at x,y:",pixel_value,center_x,center_y)
    gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
    print("grayscale :",gray_scale)
    # avg_depth = image[int(center_y), int(center_x)]

    # if avg_depth <-1:
    #     avg_depth = 1

    # Append the pixel coordinates and depth value to the stop pixel coordinates list
    
    stop_pixel_coordinates.append([center_x, 255-gray_scale,0])

# Iterate over each row in the line coordinates CSV file
for i, row in enumerate(line_coordinates_list):
    print(i)
    # Read the image file
    image_path = '/home/uthira/Documents/GitHub/MiDaS/test1752.png'  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(row[0])
    print(row[1])
    print(row[2])
    print(row[3])
    # Get the x and y coordinates
    x1, y1, x2, y2 = float(row[0])/2, float(row[1])/2, float(row[2])/2, float(row[3])/2
    print(x1,y1,x2,y2)
   

    # Calculate the average depth value
    z1 = image[int(y1), int(x1)]
    z2 = image[int(y2), int(x2)]

    pixel_value_1 = image_array[int(y1),int(x1)]
    print("pixel_value  at x,y:",pixel_value_1,x1,y1)
    gray_scale_1 = (0.2989 * pixel_value_1[0]) + (0.5870 *pixel_value_1[1]) + (0.1140 *pixel_value_1[2])
    print("grayscale :",gray_scale_1)

    pixel_value_2 = image_array[int(y2),int(x2)]
    print("pixel_value  at x,y:",pixel_value_2,x2,y2)
    gray_scale_2 = (0.2989 * pixel_value_2[0]) + (0.5870 *pixel_value_2[1]) + (0.1140 *pixel_value_2[2])
    print("grayscale :",gray_scale_2)

   

    # Append the pixel coordinates and depth value to the line pixel coordinates list
    line_pixel_coordinates.append([x1,255- gray_scale_1,0,x2,255 -gray_scale_2 ,0])
    # line_pixel_coordinates.append([x2,y2,z2])

# Create new CSV files with z values for car, person  traffic pixel coordinates
car_df = pd.DataFrame(car_pixel_coordinates, columns=['x', 'y', 'z'])
car_df.to_csv('//home/uthira/Documents/GitHub/MiDaS/csv/car_coordinates_with_z.csv', index=False)

person_df = pd.DataFrame(person_pixel_coordinates, columns=['x', 'y', 'z'])
person_df.to_csv('/home/uthira/Documents/GitHub/MiDaS/csv/person_coordinates_with_z.csv', index=False)

traffic_df = pd.DataFrame(traffic_pixel_coordinates, columns=['x', 'y', 'z'])
traffic_df.to_csv('/home/uthira/Documents/GitHub/MiDaS/csv/trafficlightwithz.csv', index=False)

stop_df = pd.DataFrame(stop_pixel_coordinates, columns=['x', 'y', 'z'])
stop_df.to_csv('/home/uthira/Documents/GitHub/MiDaS/csv/stopwithz.csv', index=False)

line_df = pd.DataFrame(line_pixel_coordinates, columns=['x1', 'y1', 'z1','x2', 'y2', 'z2'])
line_df.to_csv('/home/uthira/Documents/GitHub/MiDaS/csv/line_coordinates_with_z.csv', index=False)



# Print the first few rows of the new CSV files
print('Car coordinates with z:')
print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/car_coordinates_with_z.csv').head())

print('Person coordinates with z:')
print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/person_coordinates_with_z.csv').head())

print('traffic coordinates with z:')
print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/trafficlightwithz.csv').head())

print('stop coordinates with z:')
print(pd.read_csv('//home/uthira/Documents/GitHub/MiDaS/csv/stopwithz.csv').head())


print('line coordinates with z:')
print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/line_coordinates_with_z.csv'))



# lines_coordinates_list= load_csv('/home/uthira/Documents/GitHub/MiDaS/line_coordinates_with_z.csv')
# lines_coordinates_list = lines_coordinates_list[1:]

# lines_x1list = [float(row[0]) for row in lines_coordinates_list]
# lines_y1list = [float(row[1]) for row in lines_coordinates_list]
# lines_z1list= [float(row[2]) for row in lines_coordinates_list]
# lines_x2list = [float(row[3]) for row in lines_coordinates_list]
# lines_y2list = [float(row[4]) for row in lines_coordinates_list]
# lines_z2list= [float(row[5]) for row in lines_coordinates_list]


# print(lines_x1list)
