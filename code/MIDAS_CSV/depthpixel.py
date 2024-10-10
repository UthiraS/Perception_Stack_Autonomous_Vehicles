import numpy as np
import pandas as pd
import cv2
from csv import reader

# Read the bounding boxes csv file
bounding_boxes = pd.read_csv('bounding_boxes.csv')
pixel_coordinates =[]

def load_csv(filename):
    # Open file in read mode
    file = open(filename,"r")
    # Reading file 
    lines = reader(file)
    
    # Converting into a list 
    data = list(lines)
    return data

# Iterate over each row in the csv file
for i, row in bounding_boxes.iterrows():
    # Read the image file
    depth_map_path = '/home/uthira/Documents/GitHub/MiDaS/depth_0739.png'
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    # print("depth map size :",depth_map.size(0),depth_map.size(0))
 

    # Get the x and y coordinates
    x1, y1, x2, y2 = row['xmin']/2, row['ymin']/2, row['xmax']/2, row['ymax']/2

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Crop the image based on the bounding box coordinates
    # cropped_image = image[y1:y2, x1:x2]

    # Get the depth values for the cropped image
    # depth_value1 = depth_map[int(y1/2),int(x1/2)]
    # depth_value2 = depth_map[int(y2/2),int(x2/2)]
    # depth_values = [depth_value1,depth_value2]

    # Calculate the average depth value
    avg_depth = depth_map[int(center_y),int(center_x)]

    # Print the average depth value
    print(f"Average depth for bounding box {i+1}: {avg_depth}")

    pixel_coordinates.append([center_x,center_y,avg_depth])



df = pd.DataFrame(pixel_coordinates, columns=['x', 'y', 'z',])
df.to_csv('bounding_boxes_z.csv', index=False)


csv_filepath = "/home/uthira/Documents/GitHub/MiDaS/bounding_boxes_z.csv"
car_coordinates_list= load_csv(csv_filepath)
car_coordinates_list = car_coordinates_list[1:]

print(car_coordinates_list)

xlist = [float(row[0]) for row in car_coordinates_list]
ylist = [float(row[1]) for row in car_coordinates_list]
zlist= [float(row[2]) for row in car_coordinates_list]


print("xlist",xlist)
print("ylist",ylist)
print("zlist",zlist)
    
