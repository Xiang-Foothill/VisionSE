# Introduction
VisionSE is computer vision project from MPC Lab, Univesity of California, Berkeley, aiming for implmenting a perception algorithm that only relies on the RGB data from a windshield windshield camera for the the vehicle's 
egomotion estimation. In this project, the only information needed for the final estimation only includes the height of the camera from the ground (h), the focal length (f), and the instantaneous RGB
data collected the camera. No GPS, depth camera, or IMU is required. Also, the algorithm is completely unsupervised with no need for training with data, or preconstruction of the 
environment. Such a task is quite challenging but quite meaningful,

# 

# Advantages of this algorithm
- No need to any external assistance such as GPS
- No need for data-collection from the environment
- Absolute explainability. Unlike network or other data-driven methods that are more prevalent nowadays, the whole motion field model is completely explanable in the traditional computer-vision knowledge framework. Any defects or disadvantages of its performance can be tracked back to the theoretical level.

# Disadvantages of this algorithm
- VisionSE has very strict 
