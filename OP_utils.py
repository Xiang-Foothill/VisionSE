import numpy as np
import matplotlib.pyplot as plt
import cv2
import video_utils as vu
import data_utils as du

# This function calculates the optical flow of points selected by cv2.goodFeature 
def cv_featureLK(preImg, nextImg, deltaT, mask):
    """apply the openCV built-in function cv.calcOpticalFlowPyrLK and good to implement Lukas-Kanade 
    Method to calculate the optical-flow between two consecutive frames
    1. find all coordinates of the points with good features to be tracked
    2. calculate the optical-flow values of these points"""
    
    # set the parameters for Lukas-Kanade method
    lk_params = dict( winSize  = (8, 8),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # convert the images into grayscale
    old_gray = cv2.cvtColor(preImg, cv2.COLOR_BGR2GRAY)
    new_Gray = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)

    p0 = get_Trackpoints(old_gray, mask)

    #Calculate the values of the optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_Gray, p0, None,  **lk_params)
    """CAREFULL!!!!!! both p0 and p1 are in the form of (xs, ys) instead of (ys, xs) in the conventional representation of image coordinates in which 0th place is for vertical position and 1th place is for horizontal position"""
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    good_old, good_new = downRecenter(good_old), downRecenter(good_new)
    flow = (good_new - good_old) / deltaT

    return good_old, good_new, flow
