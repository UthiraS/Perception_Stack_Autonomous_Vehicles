# import torch
# import cv2
# import pandas as pd

# # Set NMS parameters
# conf = 0.25  # NMS confidence threshold
# iou = 0.45  # NMS IoU threshold
# agnostic = False  # NMS class-agnostic
# multi_label = True  # NMS multiple labels per box
# classes = [0,3,8,10,11,12,13,14,15]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
# max_det = 1000  # maximum number of detections per image
# amp = False

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# # Define the input images
# imgs = ['/home/uthira/Documents/GitHub/P3Data/P3Data/frame_1757.jpg']  # batch of images

# # Load the input image
# img = cv2.imread(imgs[0])
# h, w = img.shape[:2]
# print(h, w)

# # Run inference on the input image
# results = model(imgs)

# # Print and save the results
# results.print()
# results.save()  # or .show()

# # Extract the bounding box coordinates from the results
# boxes = results.xyxy[0].cpu().numpy()
# coordinates = []
# for box in boxes:
#     xmin, ymin, xmax, ymax = box[:4]
#     coordinates.append([xmin, xmax, ymin, ymax])

# # Save the bounding box coordinates to a CSV file
# df = pd.DataFrame(coordinates, columns=['xmin', 'xmax', 'ymin', 'ymax'])
# df.to_csv('bounding_boxes.csv', index=False)

# Note that you may need to modify the paths to the input image and output CSV file to match your local file system.
# Uthiralakshmi Sivaraman
# include score and separate based on label

# Sure! Here's an updated code that includes the score and separates the bounding boxes based on label:

# python

import pandas as pd
import cv2
import torch
from IPython.display import display



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
imgs = ['/home/uthira/Documents/GitHub/P3Data/P3Data/Input/test_input.jpg']  # batch of images
img = cv2.imread(imgs[0])
h, w = img.shape[:2]

print(h,w)

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

# Extract bounding boxes and labels
boxes = results.xyxy[0].cpu().numpy()
labels = results.xyxyn[0][:, -1].cpu().numpy()
scores = results.xyxyn[0][:, -2].cpu().numpy()

# Separate bounding boxes based on label
coordinates = {}
for box, label, score in zip(boxes, labels, scores):
    # if(label ==0):
    #     print("person")
    if(score>0.5):
            xmin, ymin, xmax, ymax = box[:4]
            if label not in coordinates:
                coordinates[label] = []
            coordinates[label].append([xmin, xmax, ymin, ymax, score])
    else:
        xmin, ymin, xmax, ymax = box[:4]
        if label not in coordinates:
            coordinates[label] = []
        coordinates[label].append([xmin, xmax, ymin, ymax, score])
    

# Save coordinates to CSV for each label
for label, coords in coordinates.items():
    df = pd.DataFrame(coords, columns=['xmin', 'xmax', 'ymin', 'ymax', 'score'])
    df.to_csv(f'bounding_boxes_{int(label)}.csv', index=False)
    print(f'Saved {len(coords)} bounding boxes for label {int(label)}')



