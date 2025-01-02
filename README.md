# Introduction
VisionSE is computer vision project from MPC Lab, Univesity of California, Berkeley, aiming for implmenting a perception algorithm that only relies on the RGB data from a windshield windshield camera to estimate the motion (longitudinal velocity and angular velocity).  

In this project, the only information needed for the final estimation only includes the height of the camera from the ground (h), the focal length (f), and the instantaneous RGB data collected the camera. No GPS, depth camera, or IMU is required. Also, the algorithm is completely unsupervised with no need for training with data, or preconstruction of the 
environment. Such a task is quite meaningful, since it challenges the designer to optimize the use of RGB information to an extreme level.

# Problem Formulating
Consider an ego-vehicle that moves on a flat road. Suppose that we have enough lightness in the environment to ensure a decent vision. A camera is attached to its front windshield, and the camera remains relatively still to the car. Since the road is relatively flat, we can assume that the optical axis is parallell to the ground. The camera samples images with a frequency around 24HZ. All the information we have is the image sequences, now we need to use the images to estimate the vheicle's **V_long**, and **w**. 

# Packages used
- Python 3.8.18
- numpy 1.24.3
- Opencv CV2, 4.10.0
- Matplotlib 3.7.5

# Methodology
## Optical flow and Lucas-Kanade Method
Optical flow is the apparent changes that we observe in the brightness pattern as the surrounding environment changes. In an ideal situation, optical flow should be the same as the motion field, the projection of the surrounding environment's relative motion onto the image plane.  
In this project, we choose to apply Lucas-Kanade Method to calculate optical flow, which enforces the ideal condition that the optical flow value remains constant within its small pixel neighborhood. Based on our problem formulation above, such a condition is perfectly met. 

![result1](https://github.com/user-attachments/assets/d2c99202-4b5d-4f54-be3e-ead5957d8502)

The green arrows in the image above represents the optical flow values at its correponding pixels, which is very informative about the relative motion between the surround envrionment and the ego-vehicle.

## Longuet-Higgins and Prazdny’s motion field model
Based on the assumption that the ground is flat, and the Longuet-Higgins and Prazdny's motion field model, we can now formulate the relationship between real-world egomotion $V_long$ $w$ and the pixel system motion $U_x$ $U_y$.
For all the points on the flat ground, their ego-motion and motion field satisfies the following relationship:
   
${U_x}_i = \frac{x_iy_i}{fh} * V_long + (f + \frac{x_i^2}{f}) * w$  

${U_y}_i = \frac{y_i^2}{fh} * V_long + (f + \frac{x_iy_i}{f}) * w$

In the equation above, $U_x$ and $U_y$ are the x component and y-component of the motion field value for a single pixel, indexed as i. $x_i$ and $y_i$ denote the x and y coordinate for the pixel on the image plane. $f$ is the focal length and $h$ is the height of the camera from the ground.

## Least Square Regression

By applying the Lucas-Kanade Method and its corresponding ideal assumption, we can now get the optical flow values for for a list of selected pixels at a given frame. However, all the optical values obtained here are quite noisy and may differ a lot from the real motion field values, for reasons like:
- lack of diversity of pixel intensity gradients in the selected pixel's neighborhood
- sudden change of color pattern in the environment
- Low frequency of image sampling rate  
Denote the optical flow values obtained from Lucas-Kanade Method as ${{U_x}_i'}$ and ${{U_y}_i'}$, which differs from the real motion field values by some noisy terms $\delta_x$ and $\delta_y$.
Now we can formulate two optimization problem in terms of least square:
![Optimization_formula](https://github.com/user-attachments/assets/9514b303-98a2-4515-bc8d-51fcf6dc092e)

To better understand the optimization problem above, we can interpret it as a simple learning problem, where ${{U_x}_i'}$ and  ${{U_y}_i'}$ are the labels. Terms like, $\frac{x_iy_i}{fh}$,  $\frac{x_i^2}{f})$, $\frac{y_i^2}{fh}$, and $(f + \frac{x_iy_i}{f})$ are the data poitns corresponding to the labels, and $V_long$ and $w$ are the parameters of the function to be learned.  
After finding the optimized values for ${V_long}_x$, $w_x$ (the values from the optimization problem of x-diretction motion field) and ${V_long}_y$, $w_y$ (the values from the optimiztation problem of y-direction motion field), we average the results from the two optimization problems to find the final answer.

# Advantages of this algorithm
- No need to any external assistance such as GPS
- No need for data-collection from the environment
- Absolute explainability. Unlike network or other data-driven methods that are more prevalent nowadays, the whole motion field model is completely explanable in the traditional computer-vision knowledge framework. Any defects or disadvantages of its performance can be tracked back to the theoretical level.
- 
