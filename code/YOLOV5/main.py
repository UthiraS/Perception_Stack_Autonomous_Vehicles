import os
import cv2
import pandas as pd
import cv2
import torch
from IPython.display import display


def yolomodel(imgs,filenames,valueid):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)


    # Model configuration
    # model.conf = 0.25  # NMS confidence threshold
    # model.iou = 0.45  # NMS IoU threshold
    # model.agnostic = False  # NMS class-agnostic
    # model.multi_label = True  # NMS multiple labels per box
    # model.classes = [0,3,8,10,11,12,13,14,15]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    # model.max_det = 1000  # maximum number of detections per image
    # model.amp = False

    # Images
    # imgs =[]  # batch of images
    # imgs.append(img_path)
    # img = cv2.imread(imgs[0])
    # h, w = img.shape[:2]

    # print(h,w)

    # Inference
    results = model(imgs)

    # Results
    results.print()
    # results.save()  
    boxes_values =[]
    labels_values =[]
    scores_values =[]
    # Extract bounding boxes and labels
    for i in range(len(imgs)):
        boxes = results.xyxy[i].cpu().numpy()
        labels = results.xyxyn[i][:, -1].cpu().numpy()
        scores = results.xyxyn[i][:, -2].cpu().numpy()
        boxes_values.append(boxes)
        labels_values.append(labels)
        scores_values.append(scores)
        # print(boxes)
    
    # Separate bounding boxes based on label
    n = 0
    df_car_list = []
    df_person_list = []
    df_truck_list =[]
    number = 1
    for filename,boxes, labels, scores in zip(filenames,boxes_values, labels_values, scores_values):
        coordinates = {}
        for box, label, score in zip(boxes, labels, scores):
            # if(label ==2):
            # #     print("person")
            # # if(score>0.5):
                xmin, ymin, xmax, ymax = box[:4]
                if label not in coordinates:
                    coordinates[label] = []
                coordinates[label].append([filename,xmin, xmax, ymin, ymax, score])
                print()
                # print(coordinates[label])
   
        
    
    
        # print(coordinates.items())
        for label, coords in coordinates.items():
            # if(label == 2):
            #     print("label :",label)
            #     print("coords :",coords)
            #     df_car = pd.DataFrame(coords, columns=['frame_name','xmin', 'xmax', 'ymin', 'ymax', 'score'])
            #     df_car['label'] = int(label)
            #     df_car_list.append(df_car)
            #     print(f'{len(coords)} bounding boxes saved for label {int(label)}.')
            # if(label == 0):
            #     print("label :",label)
            #     print("coords :",coords)
            #     df_person = pd.DataFrame(coords, columns=['frame_name','xmin', 'xmax', 'ymin', 'ymax', 'score'])
            #     df_person['label'] = int(label)
            #     df_person_list.append(df_person)
            #     print(f'{len(coords)} bounding boxes saved for label {int(label)}.')
            #     print("iter number :",number)
            if(label == 7):
                # print("label :",label)
                # print("coords :",coords)
                df_truck = pd.DataFrame(coords, columns=['frame_name','xmin', 'xmax', 'ymin', 'ymax', 'score'])
                df_truck['label'] = int(label)
                df_truck_list.append(df_truck)
                # print(f'{len(coords)} bounding boxes saved for label {int(label)}.')
                # print("iter number :",number)
            number += 1 


    # if df_car_list:
    #     cars_df = pd.concat(df_car_list)
    #     cars_df.to_csv('bounding_boxes_cars_'+str(valueid)+'.csv', index=False)
    # else:
    #     print("Error: No DataFrames to concatenate for cars")
    #     # Combine all dataframes into a single dataframe
    #     # combined_df = pd.concat(df_list)
    # if df_person_list:
    #     persons_df = pd.concat(df_person_list)
    #     persons_df.to_csv('bounding_boxes_persons_'+str(valueid)+'.csv', index=False)
    # else:
    #     print("Error: No DataFrames to concatenate for persons")

    if df_truck_list:
        truck_df = pd.concat(df_truck_list)
        truck_df.to_csv('/home/uthira/Documents/GitHub/Einstein_Vision/truck/bounding_boxes_truck_'+str(valueid)+'.csv', index=False)
    else:
        print("Error: No DataFrames to concatenate for persons")

    # Save the combined dataframe to a CSV file without the index
    
    
    # print('All bounding boxes saved into a single CSV file.')

def boundingboxes_frame(folder_path):
    """
    Reads every 5th frame from a folder of image files in the format frame_0001.jpg to frame_1634.jpg.
    
    Args:
        folder_path (str): The path to the folder containing the image files.
    
    Returns:
        A list of the 5th frames as NumPy arrays.
    """
    # Create a list to store the 5th frames
    frames = []
    
    valueid = 31# enter this manually due to space constrinats in CUDA


    filepaths =[]
    filenames =[]
    # Loop over the frame numbers, skipping every 5th frame
    for frame_num in range(1550,1635,5):
        # Create the filename for this frame
        filename = f"frame_{frame_num:04}.jpg"
        filepath = os.path.join(folder_path, filename)
        filepaths.append(filepath)
        filenames.append(filename)
        # print(filepath)
    yolomodel(filepaths,filenames,valueid)
        # number = number +1

        # # Read the image file into a NumPy array
        # frame = cv2.imread(filepath)

        # # Add the frame to the list
        # frames.append(frame)

    # return frames

boundingboxes_frame('/home/uthira/Documents/GitHub/Einstein_Vision/P3Data/P3Data/Sequence_Images/scene1')


 # else:
    #     xmin, ymin, xmax, ymax = box[:4]
    #     if label not in coordinates:
    #         coordinates[label] = []
    #     coordinates[label].append([xmin, xmax, ymin, ymax, score])
    

    # Save coordinates to CSV for each label
        # for label, coords in coordinates.items():
        #     df = pd.DataFrame(coords, columns=['xmin', 'xmax', 'ymin', 'ymax', 'score'])
        #     # Add the filename as a column to the DataFrame
        #     # df['filename'] = img_path
        #     # Save the DataFrame to a CSV file without the index
        #     name = "bounding_boxes_car_"+str(number)+".csv"
        #     df.to_csv(name, index=False)
        #     print(f'Saved {len(coords)} bounding boxes for label {int(label)}')
        #     number = number +1