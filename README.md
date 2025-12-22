# Driver  Warning System

This repository presents a real-time driver monitoring and warning system that detects potential driving risks by analyzing both the driver's facial state and surrounding road objects. The system is designed for real-time deployment on embedded platforms.

**For detailed methodology, system design, experiments, and results, please refer to:**  
**[`final_presentation.pdf`](./final_presentation.pdf)**

## Project Overview

The system consists of two main modules:

- **Driver Facial Analysis**
  - Emotion detection using a CNN trained on FER2013
  - Drowsiness detection based on eye-state classification and temporal rules
- **On-road Object Detection**
  - Detection of vehicles, pedestrians, and traffic-related objects
  - Distance-based risk evaluation and object tracking

Audio warnings are triggered only when dangerous conditions persist for a predefined duration to reduce false alarms.

## Key Features

- CNN-based facial emotion recognition (7 classes)
- Real-time drowsiness detection using eye-state analysis
- Object detection and tracking for on-road hazards
- Feature point tracking and Time-to-Collision (TTC) estimation
- Embedded deployment on **Jetson Xavier AGX** (~5–6 FPS)
- Combination of deep learning and classical computer vision techniques

## Technologies Used

- Python
- PyTorch
- OpenCV
- Haar Cascade (Face & Eye Detection)
- Embedded GPU deployment (Jetson Xavier AGX)

## Datasets

- FER2013 (Emotion Detection)
- Eye State Dataset (Open / Closed Eyes)
- UTA Real-Life Drowsiness Dataset (Evaluation)
- Berkeley Autonomous Driving Dataset (Object Detection)

## Author

- **Chuang Ma**

