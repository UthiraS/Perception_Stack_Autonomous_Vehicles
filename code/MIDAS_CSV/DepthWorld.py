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
traffic_coordinates_csv = 'bounding_boxes_9.csv'

car_coordinates_list = load_csv(car_coordinates_csv)[1:]  # skip header row
person_coordinates_list = load_csv(person_coordinates_csv)[1:]  # skip header row
traffic_coordinates_list = load_csv(traffic_coordinates_csv)[1:]  # skip header row

# Initialize empty lists for storing pixel coordinates and depth values
car_pixel_coordinates = []
person_pixel_coordinates = []
traffic_pixel_coordinates = []

# K = np.loadtxt('/home/uthira/Documents/GitHub/P3Data/P3Data/Calib/calibration.mat')

K = np.array([[1.5947e+03, 0, 655.2961],
              [0, 1.6077e+03, 414.3627],
              [0, 0, 1]])

# Print the matrix
print("K:", K)

R = np.array([[0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]])

T = np.array([0,0,1.4])
I = np.identity(3)
# T = T.reshape((3,1))



# Iterate over each row in the car coordinates CSV file
for i, row in enumerate(car_coordinates_list):
    # Read the image file
    image_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_map.png'  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the x and y coordinates
    x1, y1, x2, y2 = float(row[0])/2, float(row[1])/2, float(row[2])/2, float(row[3])/2
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the average depth value
    avg_depth = image[int(center_y), int(center_x)]

    if(avg_depth == 255):
        avg_depth = 1

    # u = (center_x - K[0, 2]) / K[0, 0]
    # v = (center_y - K[1, 2]) / K[1, 1]

    # Define the depth value for the pixel
    depth_value = avg_depth  # Replace this with the actual depth value

    # Convert normalized image coordinates and depth to world coordinates
    x_world = center_x * depth_value
    y_world = center_y * depth_value
    z_world = depth_value

    # Append the pixel coordinates and depth value to the car pixel coordinates list
    car_pixel_coordinates.append([x_world, y_world, z_world])
    P= np.dot(K, np.dot(R, np.hstack((I, T))))
    print("P :",P)
    print("Pinv :",np.linalg.inv(P))

    


# Iterate over each row in the person coordinates CSV file
for i, row in enumerate(person_coordinates_list):
    # Read the image file
    image_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_map.png'  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the x and y coordinates
    x1, y1, x2, y2 = float(row[0])/2, float(row[1])/2, float(row[2])/2, float(row[3])/2
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the average depth value
    avg_depth = image[int(center_y), int(center_x)]

    

    u = (center_x - K[0, 2]) / K[0, 0]
    v = (center_y - K[1, 2]) / K[1, 1]

    if(avg_depth == 255):
        avg_depth = 1

    # Define the depth value for the pixel
    depth_value = avg_depth  # Replace this with the actual depth value

    

    # Convert normalized image coordinates and depth to world coordinates
    x_world = u * depth_value
    y_world = v * depth_value
    z_world = depth_value

    # Append the pixel coordinates and depth value to the person pixel coordinates list
    person_pixel_coordinates.append([x_world, y_world, z_world])


# Iterate over each row in the traffic coordinates CSV file
for i, row in enumerate(traffic_coordinates_list):
    # Read the image file
    image_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_map.png'  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the x and y coordinates
    x1, y1, x2, y2 = float(row[0])/2, float(row[1])/2, float(row[2])/2, float(row[3])/2
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the average depth value
    avg_depth = image[int(center_y), int(center_x)]

    if(avg_depth == 255):
        avg_depth = 1

    u = (center_x - K[0, 2]) / K[0, 0]
    v = (center_y - K[1, 2]) / K[1, 1]

    # Define the depth value for the pixel
    depth_value = avg_depth  # Replace this with the actual depth value

    # Convert normalized image coordinates and depth to world coordinates
    x_world = u * depth_value
    y_world = v * depth_value
    z_world = depth_value

    # Append the pixel coordinates and depth value to the traffic pixel coordinates list
    traffic_pixel_coordinates.append([x_world, y_world, z_world])


# Iterate over each row in the line coordinates CSV file
# for i, row in enumerate(line_coordinates_list):
#     print(i)
#     # Read the image file
#     image_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_map.png'  
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     print(row[0])
#     print(row[1])
#     print(row[2])
#     print(row[3])
#     # Get the x and y coordinates
#     x1, y1, x2, y2 = float(row[0])/2, float(row[1])/2, float(row[2])/2, float(row[3])/2
#     print(x1,y1,x2,y2)
#     # center_x = (x1 + x2) / 2
#     # center_y = (y1 + y2) / 2

#     # Calculate the average depth value
#     z1 = image[int(y1), int(x1)]
#     z2 = image[int(y2), int(x2)]

#     if z1 == 255:
#         z1 = 1
#     if z2 == 255:
#         z2 = 1

#     # Append the pixel coordinates and depth value to the line pixel coordinates list
#     line_pixel_coordinates.append([x1,y1,z1,x2,y2,z2])
    # line_pixel_coordinates.append([x2,y2,z2])

# Create new CSV files with z values for car, person  traffic pixel coordinates
car_df = pd.DataFrame(car_pixel_coordinates, columns=['x', 'y', 'z'])
car_df.to_csv('car_coordinates_with_z.csv', index=False)

person_df = pd.DataFrame(person_pixel_coordinates, columns=['x', 'y', 'z'])
person_df.to_csv('person_coordinates_with_z.csv', index=False)

traffic_df = pd.DataFrame(traffic_pixel_coordinates, columns=['x', 'y', 'z'])
traffic_df.to_csv('traffic_coordinates_with_z.csv', index=False)



# Print the first few rows of the new CSV files
print('Car coordinates with z:')
print(pd.read_csv('car_coordinates_with_z.csv').head())

print('Person coordinates with z:')
print(pd.read_csv('person_coordinates_with_z.csv').head())

print('traffic coordinates with z:')
print(pd.read_csv('traffic_coordinates_with_z.csv').head())
