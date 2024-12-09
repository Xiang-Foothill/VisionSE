import numpy as np
import data_utils as du
import OP_utils as pu

def WVReg(good_old, flow, h, f):
    """apply linear regression to find the best w and v that can minimize the squared error between the optical flow measured points and the estimated line
    return the result in the order of V_tran, V_long, and w"""
    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    # prepare the data points matrix
    tran_data = - y / h
    long_data = x * y / (f * h)
    w_data = f + x ** 2 / f
    tran_data, long_data, w_data = tran_data.reshape(tran_data.shape[0], 1), long_data.reshape(long_data.shape[0], 1), w_data.reshape(w_data.shape[0], 1)
    X_data = np.concatenate((long_data, w_data), axis = 1)
    
    res_X = np.linalg.lstsq(X_data, Vx)

    return res_X[0][0], - res_X[0][1] # we should have a negative sign because our model takes downward as positive y-direction

def Kalman_filter(op_Xs, cur_X, eror):
    """key idea:
    the velocities of the car wouldn't change too much in a very short time, so the past velocities are somehow informative for our estimation of the current velocity
    combine our current estimation of the car's velocity with our knowledge of it's past velocity
    @ op_Xs: a time sequence of past estimations
    @ cur_X: our current estimation
    @ error: the error of our current estimation"""

    past_steps = 10 # the number of past steps that we want to consider


def test():
    start_frame = 0
    images, real_V, real_Omega, f, h, deltaT = du.parse_barc_data(Omega_exist=True)
    old_image = images[start_frame]
    new_image = images[start_frame + 1]
    mask = pu.G_cutoff(old_image)
    good_old, good_new , flow = pu.cv_featureLK(old_image, new_image, deltaT, mask)
    WVReg(good_old, flow, h, f)

def main():
    test()

if __name__ == "__main__":
    main()