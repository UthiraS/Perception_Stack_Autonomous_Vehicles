# Perception Stack for Autonomous Vehicles

This project implements a comprehensive perception stack for autonomous vehicles, focusing on object detection, depth estimation, and lane detection. It combines various state-of-the-art neural networks to create a Tesla-like Full Self-Driving (FSD) visualization.

## Features

- Object detection (cars, traffic signs, traffic lights, speed limit signs)
- Depth estimation
- Lane detection
- 3D visualization of detected objects

## Technologies Used

- Python
- PyTorch
- Blender (for 3D visualization)

## Neural Networks Utilized

### 1. Object Detection

- **Primary Network**: YOLOv5 (https://pytorch.org/hub/ultralytics_yolov5/)
  - Used for detecting cars, traffic signs, traffic lights, and speed limit signs
  - Bounding boxes are used to calculate midpoints of detected objects

### 2. Depth Estimation

- **Network**: MiDaS Transformer (https://github.com/jankais3r/Video-Depthify)
  - Estimates depth for the entire image
  - Average depth within bounding boxes is used for object depth estimation

### 3. Lane Detection

- **Network**: YOLO Pv2 (https://github.com/CAIC-AD/YOLOPv2)
  - Specialized for lane detection

## Implementation Details

1. **Object Detection**: 
   - YOLOv5 processes input images to detect relevant objects
   - Bounding boxes are used to calculate object midpoints

2. **Depth Estimation**:
   - MiDaS generates a depth map for the entire image
   - Average depth within object bounding boxes provides estimated object depths

3. **Lane Detection**:
   - YOLO Pv2 identifies and outlines lane markings

4. **3D Visualization**:
   - Blender is used to create a 3D representation of the detected objects and their estimated positions

## Future Improvements

- Implement instance/panoptic segmentation for more accurate object detection and depth estimation
- Integrate additional networks for specialized detection tasks (e.g., pedestrian keypoint detection)
- Enhance 3D visualization capabilities

## References

1. Lane Detection: [LaneNet](https://github.com/IrohXu/lanenet-lane-detection-pytorch)
2. Monocular Depth Estimation: [MiDaS](https://github.com/isl-org/MiDaS)
3. Object Detection (Cars, Trucks, Traffic Lights, Road Signs): 
   - [MobilenetV1-SSD](https://github.com/xiaogangLi/tensorflow-MobilenetV1-SSD)
   - [YOLOv7](https://github.com/WongKinYiu/yolov7)
4. Traffic Light Detection: [YOLOv3 for Traffic Lights](https://github.com/sovit-123/TrafficLight-Detection-Using-YOLOv3)
5. Road Sign Detection: [Road Sign Detection](https://github.com/Anantmishra1729/Road-sign-detection)
6. 3D Bounding Boxes: [YOLO3D](https://github.com/ruhyadi/YOLO3D)
7. Pedestrian Keypoint Detection: [Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

## Acknowledgements

Special thanks to Prof. Sanket for providing the front camera feed from his Tesla Model S, which was used as input data for this project.
