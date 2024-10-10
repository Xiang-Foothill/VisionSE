import numpy as np
import matplotlib.pyplot as plt
import cv2

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
def drawFlow(oldImg, old_points, new_points):
    """draw the optical_flow of old_points on oldImg"""
    thickness = 2
    color = (0, 255, 0)
    image = oldImg
    for i, op in enumerate(old_points):
        op, np = op.astype(int), new_points[i].astype(int)
        image = cv2.arrowedLine(image, op, np, color, thickness)

    cv2.imshow("Flow Diagram", image)
    cv2.waitKey(0)
    
    return image
def __main__():
    VideoToFrame()

if __name__ == "__main__":
    __main__()