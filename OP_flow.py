import numpy as np
import matplotlib.pyplot as plt
import cv2
import video_utils as vu
import data_utils as du

# This function calculates the optical flow of points selected by cv2.goodFeature 
def cv_featureLK(preImg, nextImg, deltaT):
    """apply the openCV built-in function cv.calcOpticalFlowPyrLK and good to implement Lukas-Kanade 
    Method to calculate the optical-flow between two consecutive frames
    1. find all coordinates of the points with good features to be tracked
    2. calculate the optical-flow values of these points"""

    # set the parameters for Lukas-Kanade method
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2)
    
    # find the points with good features to be tracked
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    
    # convert the images into grayscale
    old_gray = cv2.cvtColor(preImg, cv2.COLOR_BGR2GRAY)
    new_Gray = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #Calculate the values of the optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_Gray, p0, None,  **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    flow = (good_new - good_old) / deltaT
    return good_old, good_new, flow

# return: a mask array representing whether each pixel is a part of the ground or not
def G_cutoff(img: np.array) -> np.array:
    """find which part of the img is ground (True) and which part is not (False)
    this function cuts of an area from the bottom of the image, and assume all the pixels in that area are part of the ground
    return a boolean mask array"""
    H, W = img.shape[0], img.shape[1]
    # define the height and width of the ground zone
    ground_H = 180
    ground_W = 500
    bottom_center = 320

    mask = np.zeros(shape = (H, W), dtype = bool)
    mask[H - ground_H : H, int(bottom_center - ground_W / 2) : int(bottom_center + ground_W / 2)] = True
    return mask

# f_op: the optical flow function to be used
# deltaT: the time constant between two consecutive frames
def avg_Vego(f_op, preImg, nextImg, deltaT):
    """recover the egomotion from the image velocity
    Use the simplified version of plane-motion-filed equation (emega is ignored): 
    Vx = (v * x * y) / (h * f)
    Vy = (v * y ^ 2) / (h * f)
    
    find the egomotion velocity by averaging the calculated velocities of all the output flow points given by the f_op function"""
    # camera parameters
    h = 0.123 # the height of the camera from the horizontal graound 
    f = 605.5 # focal length in terms of pixels - [pixels]

    good_old, good_new, flow = f_op(preImg, nextImg, deltaT)

    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    v1 = (Vx * h * f) / (x * y)
    v2 = (Vy * h * f) / (y * y)
    V = - (np.average(v1) + np.average(v2)) / 2
    return V

def V_test():
    images, real_V = du.parse_barc_data()
    deltaT = 0.1
    op_V = []
    sample_size = 30

    for i in range(0, sample_size):
        pre_img, next_img = images[i], images[i + 1]
        V = avg_Vego(cv_featureLK, pre_img, next_img, deltaT)
        op_V.append(V)
    real_V = real_V[: sample_size]
    plt.plot(real_V, label = "real_V")
    plt.plot(op_V, label = "op_V")
    plt.legend()
    plt.show()

def test_ground_mask():
    images, real_V = du.parse_barc_data()
    index = np.random.randint(low = 0, high = images.shape[0])
    img = images[index]
    img = img.copy()
    mask = G_cutoff(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[mask] = 0
    cv2.imshow("image with the ground labeled", img)
    cv2.waitKey(0)

def __main__():
    # Frames, deltaT = vu.VideoToFrame(maxIm = 10)
    # old, new, _ = cv_featureLK(Frames[0], Frames[1], deltaT)
    # print(old.shape)
    # image = vu.drawFlow(Frames[0], old, new)
    # cv2.imwrite("result1.jpg", image)

    # V_test()
    test_ground_mask()

if __name__ == "__main__":
    __main__()