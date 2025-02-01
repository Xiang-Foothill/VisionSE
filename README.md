# Introduction
VisionSE is computer vision project from MPC Lab, Univesity of California, Berkeley, aiming for implmenting a perception algorithm that only relies on the RGB data from a windshield windshield camera to estimate the motion (longitudinal velocity and angular velocity). Such a task is quite meaningful, since it challenges the designer to optimize the use of RGB information to an extreme level. We do not want this algorithm to rely on any training (like all other learning methods) or preknowledge about the envrionment (like SLAM).

In the first half of this project, we successfully implement an algorithm based on optical flow for which the only information needed is the height of the camera from the ground (h), the focal length (f), and the instantaneous RGB data collected the camera. No GPS, depth camera, or IMU is required. Also, the algorithm is completely unsupervised with no need for training with data, or preconstruction of the environment. The key idea of such pure-vision solution is a combination of optical flow and leat-square regression based on physics model (see the methodology part for details). As we know, the measurements of optical flow is only reliable when there are enough good features to track (sharp corners with ample intensity gradients). When we do the experiment in an ideal situation (road with chessboard texture), this pure-vision algorithm turns out to perform pretty well. However, in more general road settings in which sharp-corner features appear randomly with a low frequency, this algorithm turns out to be very unstable.

To overcome this flaw(unstability when we don't have enough good features to track), in the second half of this project, we choose to fuse the pure-vision estimation with the imu estimations. When we are in a scenario where optical flow is not applicable, imu is a great backup solution in short-run. The estimation based on imu turns out to be a great complement for the pure-vision estimation.

# Problem Formulating
Consider an ego-vehicle that moves on a flat road. Suppose that we have enough lightness in the environment to ensure a decent vision. A camera is attached to its front windshield, and the camera remains relatively still to the car. Since the road is relatively flat, we can assume that the optical axis is parallell to the ground. The camera samples images with a frequency around 24HZ. All the information we have is the image sequences, now we need to use the images to estimate the vheicle's **V_long**, and **w**. 

# Methodology
### 1. Optical flow and Lucas-Kanade Method
Optical flow is the apparent changes that we observe in the brightness pattern as the surrounding environment changes. In an ideal situation, optical flow should be the same as the motion field, the projection of the surrounding environment's relative motion onto the image plane.  
In this project, we choose to apply Lucas-Kanade Method to calculate optical flow, which enforces the ideal condition that the optical flow value remains constant within its small pixel neighborhood. Based on our problem formulation above, such a condition is perfectly met. 

![result1](https://github.com/user-attachments/assets/d2c99202-4b5d-4f54-be3e-ead5957d8502)

The green arrows in the image above represents the optical flow values at its correponding pixels, which is very informative about the relative motion between the surround envrionment and the ego-vehicle.

### 2. Longuet-Higgins and Prazdny’s motion field model
Based on the assumption that the ground is flat, and the Longuet-Higgins and Prazdny's motion field model, we can now formulate the relationship between real-world egomotion $V_long$ $w$ and the pixel system motion U_x U_y.
For all the points on the flat ground, their ego-motion and motion field satisfies the following relationship:
   
![motion_model](https://github.com/user-attachments/assets/f3d745bb-4da5-4663-a56d-43e427c5442e)


In the equation above, U_x and U_y are the x component and y-component of the motion field value for a single pixel, indexed as i. x_i and y_i denote the x and y coordinate for the pixel on the image plane. f is the focal length and h is the height of the camera from the ground.

### 3. Least Square Regression

By applying the Lucas-Kanade Method and its corresponding ideal assumption, we can now get the optical flow values for for a list of selected pixels at a given frame. However, all the optical values obtained here are quite noisy and may differ a lot from the real motion field values, for reasons like:
- lack of diversity of pixel intensity gradients in the selected pixel's neighborhood
- sudden change of color pattern in the environment
- Low frequency of image sampling rate  
Denote the optical flow values obtained from Lucas-Kanade Method as U_xi and U_yi, which differs from the real motion field values by some noisy terms.
Now we can formulate two optimization problem in terms of least square:
![Optimization_formula](https://github.com/user-attachments/assets/9514b303-98a2-4515-bc8d-51fcf6dc092e)

To better understand the optimization problem above, we can interpret it as a simple learning problem, where U_xi and U_yi are the labels. All other terms can be viewed as the data poitns corresponding to the labels, and V_long and w are the parameters of the function to be learned.  
After finding the optimized values for V_longx, w_x (the values from the optimization problem of x-diretction motion field) and V_longy, w_y (the values from the optimiztation problem of y-direction motion field), we average the results from the two optimization problems to find the final answer.

# Optimization functions
### 1. Pre-filter
As mentioned above, we treat the ego-motion estimation process as a regression problem, where each pixel works as a data point and the motion values at the frame are the parameters that are going to be figured out. Then, a natural question will be: how can we remove the outliers from the data points to make such estimation as accurate as possible?  

Some pixel points in the image may provide catastrophic measurements for flow values because of the lack of texture variety in its surrounding neighborhood or a sudden change in the nearby light sources. To remove these pixels and their corresponding flows from our data, we apply the ego-motion information of the vehicles from near history (in recent 0.1 seconds) to form an approximation for the current ego-motion, Vl_past w_past. As motion state of the ego-vehicle will remain relatively stable in a very short time period, such an estimation is accurate enough to form an approximation to remove obvious outliers in the data points.  

These motion values based on history is then plugged in equation (1) and (2) to give an approximation for the flow values of each pixels at the current frame, U_xi', U_yi' With such approxmation for the flow values, we can then judge whether the measurement of flow value given by each pixel is good or not.  

![quality_factor](https://github.com/user-attachments/assets/311bd187-597b-46f6-9156-b69ed2994392)


The value Q_i is a measure of the pixel's quality. It shows how close the flow value estimated by the pixel in the current frame is to the approximation made by the history information. If this value is higher than a user-defined thershold, i.e. such a flow is way too far from the history approximation, we will exclude such a pixel from regression.
To avoid the current measurement from overly converging to the hisotry, the filtered estimation will not be used as history information for future filtering. All the history information for filtering is ego-motion measurements without any optimization method.

### 2. Past-fusion
Similar to the idea of pre-filter, we can treat the history measurements of the vehicle's ego-motion as a sensor independentfrom the current frame. Then, we can apply simple complementary sensor fusion function to the history information and the current measurement.
Again, the history information we use here is the measurements without any optimization methods.
![optical_flow_flow_chart](https://github.com/user-attachments/assets/31155f1d-4e37-42ad-b5c1-7b4e227de4da)
 
### 3. Fusion with IMU
As we can see in the experiment result above, when the car is driving in a scenario with few good features to track, optical flow becomes very usntable. We can interpret such unstability as a kind of low-frequency noise with high magnitude(extreme driving scenario appears with a low frequency, but as long as such scenario apepars, optical flow will be totally out of work). In comparison, if we only rely IMU sensor to estimate the ego-motion, we will see some high-frequency noise with low-magnitude. Such opposite natures of noises in pure-vision estimations and imu estimations make them perfect complementary sensors, which can be fused together.  
![flow_chart](https://github.com/user-attachments/assets/a4cb30b9-9273-48d9-9f99-04afcf5f85e3)

# Experiment Result
### Pure-Vision Method in ideal scenario  
Flow digram from the windshield camera  
![pure_vision_ideal_flow](https://github.com/user-attachments/assets/13334ba9-fe96-4887-a93a-e6c1a560e6b2)
  
Experiment result  
<img width="755" alt="pure_vision_ideal_result" src="https://github.com/user-attachments/assets/8acfd583-2e4a-4c7b-bee1-864a5e001f6e" />

### Pure-Vision Method in general scenario
Flow digram from the windshield camera  
![real_flow](https://github.com/user-attachments/assets/4489ecfc-066c-4713-83e5-74591e84ab39)

  
Experiment result with only the estimation from optical flow presented   
![pure_vision_real](https://github.com/user-attachments/assets/33fca6cc-30d1-4c7f-9e28-c1a4f164efdf)

Experiment result with an extra median filter applied  
![pure_vision_real_median](https://github.com/user-attachments/assets/21832bd0-3352-4600-8200-4f40f4ad815e)


### Fusion of Pure-vision Method and IMU Estimation in general scenario
Experiment result   
![fusion_real_result](https://github.com/user-attachments/assets/2044b889-4258-4ae1-9598-d4fdb674eb88)

# Advantages of this algorithm
- No need to any external assistance such as GPS
- No need for any pre-knowledge about the driving scenario
- No need for training with data
- Absolute explainability. Unlike network or other data-driven methods that are more prevalent nowadays, the whole motion field model is completely explanable in the traditional computer-vision knowledge framework. Any defects or disadvantages of its performance can be tracked back to the theoretical level.

### Contact
All the descriptions above are a brief summation of the key ideas of my work. There are many details in the codes and tedious optimization functions used in this project. If you are Interested in the details of this project, feel free to email:  
liu.yx@berkeley.edu
