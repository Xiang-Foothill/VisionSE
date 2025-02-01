# Introduction
VisionSE is an open-sourced library of a pure-vision estimator for the motion state of an ego-vehicle. By applying optical flow measurements to consecutive image frames, such an estimator can peform motion-state estimations relying only on the input RGB data from a windshield camera attached to the vehicle. If you are interested in the technical details for the implementation for this project, please visit the website: https://xiang-foothill.github.io/VisionSEWeb/webpage5.html 

# Problem Formulating
Consider an ego-vehicle that moves on a flat road. Suppose that we have enough lightness in the environment to ensure a decent vision. A camera is attached to its front windshield, and the camera remains relatively still to the car. Since the road is relatively flat, we can assume that the optical axis is parallell to the ground. The camera samples images with a frequency around 24HZ. Each time when the windshiled camera sampled a new image, now the estimator can make use of the input image to estimate the vheicle's **V_long**, and **w** (longitudinal velocity and angular velocity).  

https://github.com/user-attachments/assets/74e57e5e-d365-44eb-9577-050d621862f6

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
**STEP 3:** Initiate an estimator object which is defined. For details about the parameters needed to initiate an estimator object, please refer to the later function documents part for the Estimator Object  
```
params = <define your parameters here>
estimator = Estimators.OP_estimator(**params)
```
**STEP 4:** Use the estimate function of estimator class for further estimation. Each time when you sampled a new image from your camera, call the estimate function of the estimator object
```
Vl, w, error = estimator.estimate(input_image)
```

## `OP_estimator`
### `__init__()` paramters
- `deltaT`: the time interval in seconds between the consecutive image samples. Once such a number is set, the estimator will treat it as a default constant time between the any future pair of consecutive image samples, i.e. such an estimator treats the sampling rate of images as a fixed number.
- `h`: the height of the camera from the ground in meters.
- `f`: focal length of the camera in terms of pixels  
- `startImg`: when initiating the image, set the first image. The estimator cannot work without having the first image.

**NOTE: parameters below are used for internal optimization functions, if you are not interested in their details, please just use the default values which have been calibrated**

- `pre_filter_size`: such an estimator will make use of the information of the vehicle's motion states in near history to filter out some obvious bad optical flow measurements. This paramter decides how many previous steps will the estimator use to make filtering. For example, if it is set to 3, the estimator will use last 3 images passed into this estimator for current measurement. By default, such a value is set to 10. For more details about the implementation of this filter, please visit webpage in the introduction section.
- `pre_filter_discard`: the threshold for the prefilter to judge a pixel as a point with bad optical flow measurement. Generally, if the threshold is set lower, more pixel points will be dropped in the final estimation stage, and it will be more likely that final regression failed. By default, this is set to 8.0.
-  `past_fusion_on`: a boolean value for whether to turn on the past fusor or not. If it is true, the past fusor is turned on. If it is false, the past fusor is off. A past fusor is a fusion function that treats the ego-motion estimations from near history as an independent sensor. It will fuse such past information with current measurement. By default, such a value will be set to false.
-  `past_fusion_size`: how many last steps the past fusor is going to take. If `past_fusion_on` is set to false, such a parameter can be ignored. By default, it is set to 10.
-  `past_fusion_amplifier`: when doing past fusion, the estimator fuses the current measurement with past history based on their errors. Since the past history will always differ from the current measurement as the time progresses forward, the errors from the past history needs to be amplified. This parameter specifies how many times the errors are going to be amplified. By default, it is set to 1.5. If `past_fusion_on` is set to false, such a parameter can be ignored.

### `estimate()`
**INPUT**  
`nextImg`: the instantaneous image sampled from the windshield camera. The image should have the shape, [HEIGHT, WIDTH, CAHNNEL], where both RGB channel and GBR channels are compatible  

**OUTPUT = vl, w, error**
`vl`: measured longitudinal velocity in meter per second  
`w`: measured angular velocity in radius per second  
`error`: the error of such an estimation. It is measured through the average of all the taken pixels' minimum eigenvalues in the current frame.  

**ERROR MESSAGE**  
`Egomotion Estimation Regression FAILED !!!!!!!!!!!! f_extreme called`  
such a message will be called in the terminal if the current measurement through optical flow failed, i.e. the optical flow measurement is very unstable in the current frame. In this situation, the estimator will make use of the motion estimations in the near history as an emergency backup. The returned `vl`, `w` will be the median values among a certain number of past steps, and the error will also be calculated according to the history information.

## Test Script
In the repo, there is a CARLA test script that can directly be conducted in any CARLA driving scenario. Such a script uses the estimator to measure the motion state of a ego vehicle in CARLA driving environment. To try out the scirpt, switch to the repo's directory on your device and launch CARLA together with your driving scenario, and run the command:  
`python carlaTest.py`

# Contributor:
Yuxiang Liu, MPC Lab, Berkeley, liu.yx@berkeley.edu  
Shengfan Cao, MPC Lab, Berekeley, shengfan_cao@berkeley.edu

