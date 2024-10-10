import numpy as np
import pandas as pd
import cv2
from csv import reader
import os


def load_csv(filename):
    # Open file in read mode
    file = open(filename, "r")
    # Reading file
    lines = reader(file)
    
    # Converting into a list
    data = list(lines)
    return data

# car_coordinates_csv = '/home/uthira/Documents/GitHub/MiDaS/Overall/bounding_boxes_cars.csv'
# car_coordinates_list = load_csv(car_coordinates_csv)[1:]  # skip header row

# # print("car_coordinates_list :",car_coordinates_list)

# # set the directory where the depth images are stored
# dir_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_Images_Apr4_last/Depth_Images'
# car_pixel_coordinates =[]
# # loop over all the images in the directory
# image_list = os.listdir(dir_path)
# image_list.sort()
# for filename in image_list:
#     # check if the file is an image
#     if filename.endswith('.png') or filename.endswith('.jpg'):
#         # read the image from the file
#         image = cv2.imread(os.path.join(dir_path, filename))
#         # resize the image
#         resized = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
#         # convert the image to a NumPy array
#         image_array = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         # get the image dimensions
#         h, w = image.shape[:2]
#         # print(f'{filename} - width: {w}, height: {h}')
#         frame_num = filename.split("_")[2].split(".")[0]
#         frame_var = "frame_" + frame_num.zfill(3)
#         frame_name = frame_var+".jpg"

#         # print("frame_name :",frame_name)
        
#         for i, row in enumerate(car_coordinates_list):
#             filename,x1, x2, y1, y2,score,label = row[0],float(row[1]), float(row[2]), float(row[3]), float(row[4]),float(row[5]),row[6]
#             if filename ==frame_name:
#                 center_x = (x1 + x2) / 2
#                 center_y = (y1 + y2) / 2

        

#                 # Calculate the average depth value
#                 pixel_value = image_array[int(center_y),int(center_x)]
#                 # print("pixel_value  at x,y:",pixel_value,center_x,center_y)
#                 gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
#                 # print("grayscale :",gray_scale)
                

#                 # Append the pixel coordinates and depth value to the stop pixel coordinates list
#                 car_pixel_coordinates.append([frame_var,-center_x, 255-gray_scale,0])
#             # else :
#             #     print("frame not found")
#             #     print("filename : ",filename)
#             #     print("frame_name :",frame_name)
# # print("car_pixel_coordinates ",car_pixel_coordinates)
# car_df = pd.DataFrame(car_pixel_coordinates, columns=['frameno','x', 'y', 'z'])
# car_df.to_csv('//home/uthira/Documents/GitHub/MiDaS/csv/car_coordinates_with_z.csv', index=False)

# print('Car coordinates with z:')
# print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/car_coordinates_with_z.csv'))

# persons_coordinates_csv = '/home/uthira/Documents/GitHub/Einstein_Vision/bounding_boxes_persons.csv'
# persons_coordinates_list = load_csv(persons_coordinates_csv)[1:]  # skip header row

# # print("persons_coordinates_list :",persons_coordinates_list)

# # set the directory where the depth images are stored
# dir_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_Images_Apr4_last/Depth_Images'
# persons_pixel_coordinates =[]
# # loop over all the images in the directory
# image_list = os.listdir(dir_path)
# image_list.sort()
# for filename in image_list:
#     # check if the file is an image
#     if filename.endswith('.png') or filename.endswith('.jpg'):
#         # read the image from the file
#         image = cv2.imread(os.path.join(dir_path, filename))
#         # resize the image
#         resized = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
#         # convert the image to a NumPy array
#         image_array = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         # get the image dimensions
#         h, w = image.shape[:2]
#         # print(f'{filename} - width: {w}, height: {h}')
#         frame_num = filename.split("_")[2].split(".")[0]
#         frame_var = "frame_" + frame_num.zfill(3)
#         frame_name = frame_var+".jpg"

#         # print("frame_name :",frame_name)
        
#         for i, row in enumerate(persons_coordinates_list):
#             filename,x1, x2, y1, y2,score,label = row[0],float(row[1]), float(row[2]), float(row[3]), float(row[4]),float(row[5]),row[6]
#             if filename ==frame_name:
#                 center_x = (x1 + x2) / 2
#                 center_y = (y1 + y2) / 2

        

#                 # Calculate the average depth value
#                 pixel_value = image_array[int(center_y),int(center_x)]
#                 # print("pixel_value  at x,y:",pixel_value,center_x,center_y)
#                 gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
#                 # print("grayscale :",gray_scale)
                

#                 # Append the pixel coordinates and depth value to the stop pixel coordinates list
#                 persons_pixel_coordinates.append([frame_var,-center_x, 255-gray_scale,0])
#             # else :
#             #     print("frame not found")
#             #     print("filename : ",filename)
#             #     print("frame_name :",frame_name)
# # print("persons_pixel_coordinates ",persons_pixel_coordinates)
# persons_df = pd.DataFrame(persons_pixel_coordinates, columns=['frameno','x', 'y', 'z'])
# persons_df.to_csv('//home/uthira/Documents/GitHub/MiDaS/csv/persons_coordinates_with_z.csv', index=False)

# print('persons coordinates with z:')
# print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/persons_coordinates_with_z.csv'))


# roadsigns_coordinates_csv = '/home/uthira/Documents/GitHub/MiDaS/Overall/roadsign.csv'
# roadsigns_coordinates_list = load_csv(roadsigns_coordinates_csv)[1:]  # skip header row

# # print("roadsigns_coordinates_list :",roadsigns_coordinates_list)

# # set the directory where the depth images are stored
# dir_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_Images_Apr4_last/Depth_Images'
# roadsigns_pixel_coordinates =[]
# # loop over all the images in the directory
# image_list = os.listdir(dir_path)
# image_list.sort()
# for filename in image_list:
#     # check if the file is an image
#     if filename.endswith('.png') or filename.endswith('.jpg'):
#         # read the image from the file
#         image = cv2.imread(os.path.join(dir_path, filename))
#         # resize the image
#         resized = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
#         # convert the image to a NumPy array
#         image_array = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         # get the image dimensions
#         h, w = image.shape[:2]
#         # print(f'{filename} - width: {w}, height: {h}')
#         frame_num = filename.split("_")[2].split(".")[0]
#         frame_var = "frame_" + frame_num.zfill(3)
#         frame_name = frame_var

#         # print("frame_name :",frame_name)
        
#         for i, row in enumerate(roadsigns_coordinates_list):
#             filename,score_label,x1, y1, x2, y2 = row[0],row[1],float(row[2]), float(row[3]), float(row[4]), float(row[5])
#             split_string = score_label.split(" ")

#             variable = split_string[0]
#             score = float(split_string[1])
#             # print("variable :",variable)
#             if variable != "trafficlight":

           
#                 if filename ==frame_name:
#                     center_x = (x1 + x2) / 2
#                     center_y = (y1 + y2) / 2

#                     # Calculate the average depth value
#                     pixel_value = image_array[int(center_y),int(center_x)]
#                     # print("pixel_value  at x,y:",pixel_value,center_x,center_y)
#                     gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
#                     # print("grayscale :",gray_scale)
                    

#                     # Append the pixel coordinates and depth value to the stop pixel coordinates list
#                     roadsigns_pixel_coordinates.append([frame_var,-center_x, 255-gray_scale,0])
           
# roadsigns_df = pd.DataFrame(roadsigns_pixel_coordinates, columns=['frameno','x', 'y', 'z'])
# roadsigns_df.to_csv('//home/uthira/Documents/GitHub/MiDaS/csv/roadsigns_coordinates_with_z.csv', index=False)

# print('roadsigns coordinates with z:')
# print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/roadsigns_coordinates_with_z.csv'))


# traffic_coordinates_csv = '/home/uthira/Documents/GitHub/MiDaS/Overall/roadsign.csv'
# traffic_coordinates_list = load_csv(traffic_coordinates_csv)[1:]  # skip header row

# # print("traffic_coordinates_list :",traffic_coordinates_list)

# # set the directory where the depth images are stored
# dir_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_Images_Apr4_last/Depth_Images'
# traffic_pixel_coordinates =[]
# # loop over all the images in the directory
# image_list = os.listdir(dir_path)
# image_list.sort()
# for filename in image_list:
#     # check if the file is an image
#     if filename.endswith('.png') or filename.endswith('.jpg'):
#         # read the image from the file
#         image = cv2.imread(os.path.join(dir_path, filename))
#         # resize the image
#         resized = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
#         # convert the image to a NumPy array
#         image_array = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         # get the image dimensions
#         h, w = image.shape[:2]
#         # print(f'{filename} - width: {w}, height: {h}')
#         frame_num = filename.split("_")[2].split(".")[0]
#         frame_var = "frame_" + frame_num.zfill(3)
#         frame_name = frame_var

#         # print("frame_name :",frame_name)
        
#         for i, row in enumerate(traffic_coordinates_list):
#             filename,score_label,x1, y1, x2, y2 = row[0],row[1],float(row[2]), float(row[3]), float(row[4]), float(row[5])
#             split_string = score_label.split(" ")

#             variable = split_string[0]
#             score = float(split_string[1])
#             # print("variable :",variable)
#             if variable == "trafficlight":

           
#                 if filename ==frame_name:
#                     center_x = (x1 + x2) / 2
#                     center_y = (y1 + y2) / 2

#                     # Calculate the average depth value
#                     pixel_value = image_array[int(center_y),int(center_x)]
#                     # print("pixel_value  at x,y:",pixel_value,center_x,center_y)
#                     gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
#                     # print("grayscale :",gray_scale)
                    

#                     # Append the pixel coordinates and depth value to the stop pixel coordinates list
#                     traffic_pixel_coordinates.append([frame_var,-center_x, 255-gray_scale,-center_y])
           
# traffic_df = pd.DataFrame(traffic_pixel_coordinates, columns=['frameno','x', 'y', 'z'])
# traffic_df.to_csv('//home/uthira/Documents/GitHub/MiDaS/csv/traffic_coordinates_with_z.csv', index=False)

# print('traffic coordinates with z:')
# print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/traffic_coordinates_with_z.csv'))


truck_coordinates_csv = '/home/uthira/Documents/GitHub/Einstein_Vision/truck/bounding_boxes_truck.csv'
truck_coordinates_list = load_csv(truck_coordinates_csv)[1:]  # skip header row

# print("truck_coordinates_list :",truck_coordinates_list)

# set the directory where the depth images are stored
dir_path = '/home/uthira/Documents/GitHub/MiDaS/Depth_Images_Apr4_last/Depth_Images'
truck_pixel_coordinates =[]
# loop over all the images in the directory
image_list = os.listdir(dir_path)
image_list.sort()
for filename in image_list:
    # check if the file is an image
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # read the image from the file
        image = cv2.imread(os.path.join(dir_path, filename))
        # resize the image
        resized = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
        # convert the image to a NumPy array
        image_array = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # get the image dimensions
        h, w = image.shape[:2]
        # print(f'{filename} - width: {w}, height: {h}')
        frame_num = filename.split("_")[2].split(".")[0]
        frame_var = "frame_" + frame_num.zfill(3)
        frame_name = frame_var+".jpg"

        # print("frame_name :",frame_name)
        
        for i, row in enumerate(truck_coordinates_list):
            filename,x1, x2, y1, y2,score,label = row[0],float(row[1]), float(row[2]), float(row[3]), float(row[4]),float(row[5]),row[6]
            if filename ==frame_name:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

        

                # Calculate the average depth value
                pixel_value = image_array[int(center_y),int(center_x)]
                # print("pixel_value  at x,y:",pixel_value,center_x,center_y)
                gray_scale = (0.2989 * pixel_value[0]) + (0.5870 *pixel_value[1]) + (0.1140 *pixel_value[2])
                # print("grayscale :",gray_scale)
                

                # Append the pixel coordinates and depth value to the stop pixel coordinates list
                truck_pixel_coordinates.append([frame_var,-center_x, 255-gray_scale,0])
            # else :
            #     print("frame not found")
            #     print("filename : ",filename)
            #     print("frame_name :",frame_name)
# print("truck_pixel_coordinates ",truck_pixel_coordinates)
truck_df = pd.DataFrame(truck_pixel_coordinates, columns=['frameno','x', 'y', 'z'])
truck_df.to_csv('//home/uthira/Documents/GitHub/MiDaS/csv/truck_coordinates_with_z.csv', index=False)

print('truck coordinates with z:')
print(pd.read_csv('/home/uthira/Documents/GitHub/MiDaS/csv/truck_coordinates_with_z.csv'))
