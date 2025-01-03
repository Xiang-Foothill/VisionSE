# Introduction
VisionSE is computer vision project from MPC Lab, Univesity of California, Berkeley, aiming for implmenting a perception algorithm that only relies on the RGB data from a windshield windshield camera to estimate the motion (longitudinal velocity and angular velocity). Such a task is quite meaningful, since it challenges the designer to optimize the use of RGB information to an extreme level. We do not want this algorithm to rely on any training (like all other learning methods) or preknowledge about the envrionment (like SLAM).

In the first half of this project, we successfully implement an algorithm based on optical flow for which the only information needed is the height of the camera from the ground (h), the focal length (f), and the instantaneous RGB data collected the camera. No GPS, depth camera, or IMU is required. Also, the algorithm is completely unsupervised with no need for training with data, or preconstruction of the environment. The key idea of such pure-vision solution is a combination of optical flow and leat-square regression based on physics model (see the methodology part for details). As we know, the measurements of optical flow is only reliable when there are enough good features to track (sharp corners with ample intensity gradients). When we do the experiment in an ideal situation (road with chessboard texture), this pure-vision algorithm turns out to perform pretty well. However, in more general road settings in which sharp-corner features appear randomly with a low frequency, this algorithm turns out to be very unstable.

To overcome this flaw(unstability when we don't have enough good features to track), in the second half of this project, we choose to fuse the pure-vision estimation with the imu estimations. When we are in a scenario where optical flow is not applicable, imu is a great backup solution in short-run. The estimation based on imu turns out to be a great complement for the pure-vision estimation.

# Problem Formulating
Consider an ego-vehicle that moves on a flat road. Suppose that we have enough lightness in the environment to ensure a decent vision. A camera is attached to its front windshield, and the camera remains relatively still to the car. Since the road is relatively flat, we can assume that the optical axis is parallell to the ground. The camera samples images with a frequency around 24HZ. All the information we have is the image sequences, now we need to use the images to estimate the vheicle's **V_long**, and **w**. 

# Packages used
- Python 3.8.18
- numpy 1.24.3
- Opencv CV2, 4.10.0
- Matplotlib 3.7.5

# Files and their function
There are seven scripts in this repo:
- **carlaGym.py** : a script that interacts with CARLA simulation environment to collect test data of a specific format. The data collected can then be used to do experiments in the test.py script. Please check the comments in this file to see details about how to use it.
- **data_utils.py** : a collection of all data utility functions. Note that all the functions in this script are written to preprocess the specific format of data collected in the carlaGym.py file. Please see the comments in this file to see details about how to use it.
- **Ego-motion.py** : a collection of ego-motion estimation functions and some optimization functions (all kinds of filters). All functions here are designed to provide estimations for ego-motion based on given optical-flow values.
- **op_development.py**: All the functions here are written during the process of development and debugging. Please ignore this file.
- **OP_utils.py** : a collection of functions used for calculating the optical flow values based on RGB data.
- **test.py** : a collection of test functions. Please see the comments in this file to see details about how to use it.
- **vedio_utils.py** : a collection of video and image utility functions, including functions used to plot flow diagrams, to show images, and to plot experiment results.

# Experiment Result
### Pure-Vision Method in ideal scenario  
Flow digram from the windshield camera  
![pure_vision_ideal_flow](https://github.com/user-attachments/assets/13334ba9-fe96-4887-a93a-e6c1a560e6b2)
  
Experiment result  
<img width="755" alt="pure_vision_ideal_result" src="https://github.com/user-attachments/assets/8acfd583-2e4a-4c7b-bee1-864a5e001f6e" />

### Pure-Vision Method in general scenario
Flow digram from the windshield camera  
![real_flow](https://github.com/user-attachments/assets/4489ecfc-066c-4713-83e5-74591e84ab39)

  
Experiment result  
![pure_vision_real](https://github.com/user-attachments/assets/b296c0fa-82f8-4720-9e7e-a8e3feeb190c)


### Fusion of Pure-vision Method and IMU Estimation in general scenario
Experiment result   
![fusion_real_result](https://github.com/user-attachments/assets/2044b889-4258-4ae1-9598-d4fdb674eb88)


# How to redo the experiment
To redo the simulation experiment on your device, please save this repo as a source folder on any position you like. Then create a folder in a position that is parallel to the whole source folder, and name such folder as "VideoSet".
![VideoSet_demo](https://github.com/user-attachments/assets/7e0c165b-c8f0-4bca-9624-da6727d58653)

Such a folder is used to save the simulation data packages (images, real longitudinal velocities, real angular velocites and measurements by imu) used for the experiment. To download exact the same data packges that we use for our experiment, please open the google drive link: https://drive.google.com/drive/folders/1Nxq7R_4_NG3K1WwQHacCBG2baGkH8K1x?usp=drive_link. Please place all the data .pkl files in the VideoSet folder. Also, you can use the carlaGym.py script to collect data in any CARLA driving scenario that you specify. Still, please make sure you place the collected data files in the "VideoSet" folder.
Then, open the test.py script and choose the test function that you want to use. You may test the accuracy of our estimations when only rgb (imu) data is used, when only imu is used, and when rgb data and imu data are fused together. Then, run the test.py script and you will see the experiment result.

# Methodology
### 1. Optical flow and Lucas-Kanade Method
Optical flow is the apparent changes that we observe in the brightness pattern as the surrounding environment changes. In an ideal situation, optical flow should be the same as the motion field, the projection of the surrounding environment's relative motion onto the image plane.  
In this project, we choose to apply Lucas-Kanade Method to calculate optical flow, which enforces the ideal condition that the optical flow value remains constant within its small pixel neighborhood. Based on our problem formulation above, such a condition is perfectly met. 

![result1](https://github.com/user-attachments/assets/d2c99202-4b5d-4f54-be3e-ead5957d8502)

The green arrows in the image above represents the optical flow values at its correponding pixels, which is very informative about the relative motion between the surround envrionment and the ego-vehicle.

### 2. Longuet-Higgins and Prazdny’s motion field model
Based on the assumption that the ground is flat, and the Longuet-Higgins and Prazdny's motion field model, we can now formulate the relationship between real-world egomotion $V_long$ $w$ and the pixel system motion $U_x$ $U_y$.
For all the points on the flat ground, their ego-motion and motion field satisfies the following relationship:
   
${U_x}_i = \frac{x_iy_i}{fh} * V_long + (f + \frac{x_i^2}{f}) * w$  

${U_y}_i = \frac{y_i^2}{fh} * V_long + (f + \frac{x_iy_i}{f}) * w$

In the equation above, $U_x$ and $U_y$ are the x component and y-component of the motion field value for a single pixel, indexed as i. $x_i$ and $y_i$ denote the x and y coordinate for the pixel on the image plane. $f$ is the focal length and $h$ is the height of the camera from the ground.

### 3. Least Square Regression

By applying the Lucas-Kanade Method and its corresponding ideal assumption, we can now get the optical flow values for for a list of selected pixels at a given frame. However, all the optical values obtained here are quite noisy and may differ a lot from the real motion field values, for reasons like:
- lack of diversity of pixel intensity gradients in the selected pixel's neighborhood
- sudden change of color pattern in the environment
- Low frequency of image sampling rate  
Denote the optical flow values obtained from Lucas-Kanade Method as ${{U_x}_i'}$ and ${{U_y}_i'}$, which differs from the real motion field values by some noisy terms $\delta_x$ and $\delta_y$.
Now we can formulate two optimization problem in terms of least square:
![Optimization_formula](https://github.com/user-attachments/assets/9514b303-98a2-4515-bc8d-51fcf6dc092e)

To better understand the optimization problem above, we can interpret it as a simple learning problem, where ${{U_x}_i'}$ and  ${{U_y}_i'}$ are the labels. Terms like, $\frac{x_iy_i}{fh}$,  $\frac{x_i^2}{f})$, $\frac{y_i^2}{fh}$, and $(f + \frac{x_iy_i}{f})$ are the data poitns corresponding to the labels, and $V_long$ and $w$ are the parameters of the function to be learned.  
After finding the optimized values for ${V_long}_x$, $w_x$ (the values from the optimization problem of x-diretction motion field) and ${V_long}_y$, $w_y$ (the values from the optimiztation problem of y-direction motion field), we average the results from the two optimization problems to find the final answer.

### 4. Fusion with IMU
As we can see in the experiment result above, when the car is driving in a scenario with few good features to track, optical flow becomes very usntable. We can interpret such unstability as a kind of low-frequency noise with high magnitude(extreme driving scenario appears with a low frequency, but as long as such scenario apepars, optical flow will be totally out of work). In comparison, if we only rely IMU sensor to estimate the ego-motion, we will see some high-frequency noise with low-magnitude. Such opposite natures of noises in pure-vision estimations and imu estimations make them perfect complementary sensors, which can be fused together.  
![flow_chart](https://github.com/user-attachments/assets/a4cb30b9-9273-48d9-9f99-04afcf5f85e3)

# Advantages of this algorithm
- No need to any external assistance such as GPS
- No need for any pre-knowledge about the driving scenario
- No need for training with data
- Absolute explainability. Unlike network or other data-driven methods that are more prevalent nowadays, the whole motion field model is completely explanable in the traditional computer-vision knowledge framework. Any defects or disadvantages of its performance can be tracked back to the theoretical level.

### Contact
All the descriptions above are a brief summation of the key ideas of my work. There are many details in the codes and tedious optimization functions used in this project. If you are Interested in the details of this project, feel free to email:  
liu.yx@berkeley.edu
