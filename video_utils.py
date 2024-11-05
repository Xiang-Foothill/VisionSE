import numpy as np
import matplotlib.pyplot as plt
import cv2
import data_utils as du
import math

# PATH: the path to the stored mp4 file
# FRAMES: a list of numpy arrays
# DELTAT: the time interval between two consecutive frames
#MAXIM: the maxmimum number of images to be included in the returned frame list
def VideoToFrame(path = "e:/VisionSE/VideoSet/testVideo1.mp4", maxIm = 400):
    """Apply cv2 VideoCapture Project to transform a mp4 video into
    a list of frames, where each frame is represented by a numpy array"""

    cap = cv2.VideoCapture(path)

    if(cap.isOpened() == False):
        print("!!!! Fail to open the video file !!!!")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    deltaT = 1 / fps
    Frames = []
    count = 0

    while(cap.isOpened() and count < maxIm):
        ret, frame = cap.read()

        if(ret):
            Frames.append(frame)
        count += 1
    return Frames, deltaT

# old_points: the 2D coordinates of the points whose optical flow to be drawn
# new_points: the 2D coordinates of old_points after the shifting of their optical flow
def drawFlow(oldImg, old_points, new_points, title = "flow Diagram", pause = True, show = True):
    """draw the optical_flow of old_points on oldImg"""
    thickness = 2
    color = (0, 255, 0)
    image = oldImg
    for i, op in enumerate(old_points):
        op, np = op.astype(int), new_points[i].astype(int)
        image = cv2.arrowedLine(image, op, np, color, thickness)

    if show:
        cv2.imshow(title, image)

    if pause:
        cv2.waitKey(0)

    return image

# use this display function as priority, since it compiles with pixel type of higher bits -> clearer image
# This method shows all the image in the image_list in a single large window
# image_list is a dictionary with the names of the image as keys
# image pixels may have unit of 64 bytes
def show_IM_window(image_list, is_GrayScale = True):
    n = math.ceil(math.sqrt(len(image_list) + 1))
    m = math.ceil((len(image_list) + 1) / n)
    fig, axes = plt.subplots(m, n, figsize = (16, 9))
    
    count = 0
    for key in image_list:
        ax = axes[count // n][count % n]
        if not is_GrayScale:
            ax.imshow(image_list[key])
        else:
            image = image_list[key]
            ax.imshow(image, cmap = 'gray')
        ax.set_title(key)
        count += 1
        ax.axis("off")

def npArrowFlow(oldImg, old_points, new_points):
    """generate the diagram with flow-arrow, make sure that the image is in the format of numpy: RGB format"""
    thickness = 2
    color = (0, 255, 0)
    image = oldImg
    for i, op in enumerate(old_points):
        op, np = op.astype(int), new_points[i].astype(int)
        image = cv2.arrowedLine(image, op, np, color, thickness)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
    
def __main__():
    VideoToFrame()

if __name__ == "__main__":
    __main__()