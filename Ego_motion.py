import numpy as np


def WVReg(good_old, flow, h, f):
    """apply linear regression to find the best w and v that can minimize the squared error between the optical flow measured points and the estimated line"""
    