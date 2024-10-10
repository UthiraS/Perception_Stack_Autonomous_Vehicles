# Perception_Stack_Autonomous_Vehicles

 
Networks utilized:

1. Object detection (cars, traffic signs, traffic lights, speed limit sign):

We used this one for object detection: https://pytorch.org/hub/ultralytics_yolov5/

We found midpoints of the cars, traffic signs, traffic lights using the bounding boxes obtained from this network by feeding it the images.

However, we plan on using instance/panoptic segmentation network in hopes that it will provide better results for depth estimation (using average depth in the segmented part of the image)

2. Depth estimation:

We used transformer based midas: https://github.com/jankais3r/Video-Depthify

We took the bounding boxes from object detection and found average depths in the same regions as those of bounding boxes to obtain estimated depths. However, we expect to use instance segmentation instead of object detection for better results.

3. Lane detection:

YOLO Pv2 : https://github.com/CAIC-AD/YOLOPv2

Detected the lanes through this network.

Thank you!
 
[1] Lane Detection: https://github.com/IrohXu/lanenet-lane-detection-pytorch

[2] Monocular Depth Estimation: https://github.com/isl-org/MiDaS

[3] Object Detection: Cars, Trucks, Traffic Lights, Road Signs:
https://github.com/xiaogangLi/tensorflow-MobilenetV1-SSD

[4] Object Detection: Cars, Trucks, Traffic Lights, Road Signs:
https://github.com/WongKinYiu/yolov7

[5] Object Detection: Traffic Lights: https://github.com/sovit-123/TrafficLight-Detection-Using-YOLOv3

[6] Object Detection: Road Signs: https://github.com/Anantmishra1729/Road-sign-detection

[7] YOLO 3-D bounding boxes: https://github.com/ruhyadi/YOLO3D

[8] Pedestrian keypoint detection: https://github.com/ZheC/Realtime MultiPerson Pose Estimation



