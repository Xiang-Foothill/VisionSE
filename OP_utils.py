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
    # p0 = add_track_points(p0)

    #Calculate the values of the optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_Gray, p0, None,  **lk_params)
    """CAREFULL!!!!!! both p0 and p1 are in the form of (xs, ys) instead of (ys, xs) in the conventional representation of image coordinates in which 0th place is for vertical position and 1th place is for horizontal position"""
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    good_old, good_new = downRecenter(good_old), downRecenter(good_new)
    flow = (good_new - good_old) / deltaT

    return good_old, good_new, flow

def get_Trackpoints(img, ground_mask):
    # specify the function used to detect the ground
    # f_ground = G_cutoff

    # find the points with good features to be tracked
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.31,
                       minDistance = 7,
                       blockSize = 7)
    
    #use the ground_function to get the ground mask
    # ground_mask = f_ground(img).astype(np.uint8)
    ground_mask = ground_mask.astype(np.uint8)
    # ground_mask = None
    p0 = cv2.goodFeaturesToTrack(img, mask = ground_mask, **feature_params)

    if p0 is None:
        #deal with the case when p0 is None, i.e. no good features are detected, just return random points on the ground
        # P = 0.1
        # rand_points = np.random.uniform(low = 0, high = 1, size = (img.shape[0], img.shape[1]))
        # rand_mask = rand_points > P
        # mask = np.logical_and(rand_mask, ground_mask)
        # return np.float32(np.argwhere(mask == True))
        return cv2.goodFeaturesToTrack(img, mask = None, **feature_params)
    else:
        return p0

def add_track_points(p0):
    """in real life situations, there might not be that many good features to track on the ground, which may lead to a very small number of available data points that can be used for regression
    this function aims to add data points, so that more optical values can be found and used for regression"""

    threshold = 80 # the threshold over which we treat the number of data points as enough

    if p0.shape[0] >= threshold:
        return p0 # if we have enough data points for regression return p0 directly
    
    # decide the upper and lower boundary in which the points can be generated
    x_low, x_high, y_low, y_high = 10, 630, 250, 470 # be careful that such coordinates refer to the coordinates before the DOWNCENTER function is applied

    np.random.seed(0)
    # now add points randomly
    for _ in range(threshold - p0.shape[0]):
        x_new, y_new = np.random.randint(low = x_low, high = x_high), np.random.randint(low = y_low, high = y_high)
        p0 = np.concatenate((p0, np.asarray([[[x_new, y_new]]])), axis = 0)

    p0 = p0.astype(np.float32) # the above python opretions change the data type for the coordinate array, reshape it back to float 32

    return p0

# coord: the coordinates of the pixels to be changed
def downRecenter(coord):
    """relocate the origin of the pixel system to the center of the image plane
    the original pixel system takes the top-left corner as (0, 0)
    Different from the recenter function, this function relocates coordinates so that the positive y-direction is downward"""
    new_coord = coord.copy()
    W = 640
    H = 480

    new_coord[:, 0] = new_coord[:, 0] - W * 0.5 # change the x-coordinates
    new_coord[:, 1] = new_coord[:, 1] - H * 0.5 # change the y-coordinates

    return new_coord

# return: a mask array representing whether each pixel is a part of the ground or not
def G_cutoff(img: np.array) -> np.array:
    """find which part of the img is ground (True) and which part is not (False)
    this function cuts of an area from the bottom of the image, and assume all the pixels in that area are part of the ground
    return a boolean mask array"""
    H, W = img.shape[0], img.shape[1]
    # define the height and width of the ground zone
    ground_H = 300
    ground_W = 630
    bottom_center = W * 0.5
    ground_start = 0

    mask = np.zeros(shape = (H, W), dtype = bool)
    # mask[:200, :100] = True
    mask[H - ground_H - ground_start: H - ground_start, int(bottom_center - ground_W / 2) : int(bottom_center + ground_W / 2)] = True
    return mask