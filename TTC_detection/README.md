## Vehicle Detection and Time-to-Collision (TTC) Estimation

To enable dynamic risk awareness, the system incorporates **vehicle detection** and **motion-based Time-to-Collision (TTC) estimation** using inter-frame visual cues.

### Vehicle Detection

Vehicles are detected using a deep learning–based object detector (YOLO), which provides real-time bounding boxes for objects of interest.  
Detected vehicle regions serve as the basis for subsequent motion analysis and collision risk estimation.

### Feature-Based TTC Estimation

For each detected vehicle, **Time-to-Collision (TTC)** is estimated by analyzing **feature point motion across consecutive frames**:

- Feature points are extracted within each vehicle bounding box
- Feature correspondences are established between consecutive frames
- TTC is estimated from the relative change (expansion) of feature point distances over time

This approach assumes approximately constant velocity motion and enables the system to determine whether a vehicle is **approaching the camera**, even when absolute depth measurements are noisy or unreliable.

### Safety-Aware Integration

The estimated TTC is combined with distance information to assess collision risk:

- Objects with **smaller TTC values** are treated as higher-priority hazards
- TTC helps distinguish **static obstacles** from **approaching vehicles**
- Safety-critical objects trigger earlier and more prominent audio alerts
