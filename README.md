# Introduction
VisionSE is an open-sourced library of a pure-vision estimator for the motion state of an ego-vehicle. By applying optical flow measurements to consecutive image frames, such an estimator can peform motion-state estimations relying only on the input RGB data from a windshield camera attached to the vehicle. If you are interested in the technical details for the implementation for this project, please visit the website: https://xiang-foothill.github.io/VisionSEWeb/webpage5.html 

# Problem Formulating
Consider an ego-vehicle that moves on a flat road. Suppose that we have enough lightness in the environment to ensure a decent vision. A camera is attached to its front windshield, and the camera remains relatively still to the car. Since the road is relatively flat, we can assume that the optical axis is parallell to the ground. The camera samples images with a frequency around 24HZ. All the information we have is the image sequences, now the estimator can make use of these images to estimate the vheicle's **V_long**, and **w** (longitudinal velocity and angular velocity).

# Packages Required
The scripts in this library made use of the following packages:
- Python 3.8.18
- numpy 1.24.3
- Opencv CV2, 4.10.0
- Matplotlib 3.7.5  
Please make sure you have all the corresponding packages installed with compatible versions before you make use of estimator in this library  

# How to use this estimator
**STEP 1:** Please clone this repo to a directory at your working space  
**STEP 2:** import the estimator from the directory in which you save this repo
```
sys.path.insert(0, "<the directory in which you save this repo>")    
import Estimators
```
**STEP 3:** Initiate an estimator object which is defined. For details about the parameters needed to initiate an estimator object, please refer to the later function documents part   
```
params = <define your parameters here>
estimator = Estimators.OP_estimator(**params)
```
**STEP 4:** Use the estimate function of estimator class for further estimation
```
Vl, w, error = estimator.estimate(input_image)
```

# Functions Document





