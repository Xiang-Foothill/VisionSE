import numpy as np
import data_utils as du
import OP_utils as pu

def WVReg(good_old, flow, h, f):
    """apply linear regression to find the best w and v that can minimize the squared error between the optical flow measured points and the estimated line
    return the result in the order of V_tran, V_long, and w"""
    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    # prepare the data points matrix
    X_tran_data = - y / h
    X_long_data = x * y / (f * h)
    X_w_data = f + x ** 2 / f
    Y_long_data = y ** 2 / (f * h)
    Y_w_data = (x * y) / f

    X_tran_data, X_long_data, X_w_data = X_tran_data.reshape(X_tran_data.shape[0], 1), X_long_data.reshape(X_long_data.shape[0], 1), X_w_data.reshape(X_w_data.shape[0], 1)
    Y_long_data, Y_w_data = Y_long_data.reshape(Y_long_data.shape[0], 1), Y_w_data.reshape(Y_w_data.shape[0], 1)
    X_data = np.concatenate((X_long_data, X_w_data), axis = 1)
    Y_data = np.concatenate((Y_long_data, Y_w_data), axis = 1)

    res_X = np.linalg.lstsq(X_data, Vx)
    res_Y = np.linalg.lstsq(Y_data, Vy)

    V_long_X, w_X, Error_X = res_X[0][0], - res_X[0][1], res_X[1][0] # we should have a negative sign for the angular velocity because our model takes downward as positive y-direction
    V_long_Y, w_Y, Error_Y = res_Y[0][0], - res_Y[0][1], res_Y[1][0]

    V_long, w, error = (V_long_X + V_long_Y) / 2, (w_X + w_Y) / 2, (Error_X + Error_Y) / 2
    return  V_long, w, error

def Kalman_filter(op_Xs, op_errors, cur_X, error):
    """key idea:
    the velocities of the car wouldn't change too much in a very short time, so the past velocities are somehow informative for our estimation of the current velocity
    combine our current estimation of the car's velocity with our knowledge of it's past velocity
    @ op_Xs: a time sequence of past estimations
    @ op_errors: the errors in the past history
    @ cur_X: our current estimation
    @ error: the error of our current estimation"""
    past_amplifier = 1.0 # past informations may have some errors regarding the current time no matter what, amplify the error given by the past time measurement by this factor
    past_steps = 5 # the number of past steps that we want to consider
    past_len = len(op_Xs)

    if past_len == 0:
        return cur_X # if we are at the very first step, return the original estimation directly, there is no way to use past information for estimation
    
    past_steps = min(past_steps, past_len)
    past_Xs, past_Errors = op_Xs[- past_steps : ], op_errors[- past_steps : ]

    # now take the average of past measurements and past errors
    past_X, past_Error = np.average(past_Xs), np.average(past_Errors)
    past_Error *= past_amplifier

    Kalman_gain = error / (error + past_Error)
    res_X = cur_X + Kalman_gain * (past_X - cur_X)
    return res_X

def pre_filter(good_old, flow, h, f, op_Vl, op_w):
    """key idea:
    use the information from past time (op_Vl, op_w) to filter out obvious outliers in the data, since the velocities of the car cannot change too much in very limited time
    for a point with Vx, Vy, and pre_Vx, pre_Vy
    its PREFACTOR = abs((Vx - pre_Vx) / pre_Vx ) + abs((Vy - pre_Vy) / pre_Vy)

    @ return: flow vector without obvious outliers"""
    discard_threshold = 3.0 # the threshold for discard a point, if a point's preFactor is higher than this threshold, discard it

    past_steps = 5  # the length of the horizon of how long we want to look back
    past_len = len(op_Vl)

    if past_len == 0:
        return good_old, flow # if we are at the very start, this function is not applicable, return good_old, and flow directly
    
    # obtain past information
    past_steps = min(past_len, past_steps)
    past_Vl, past_w = np.average(op_Vl[- past_steps : ]), np.average(op_w[- past_steps : ])
    pre_VW = np.asarray([[past_Vl], [past_w]])

    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]

    # prepare the data points matrix
    X_tran_data = - y / h
    X_long_data = x * y / (f * h)
    X_w_data = f + x ** 2 / f
    Y_long_data = y ** 2 / (f * h)
    Y_w_data = (x * y) / f

    X_tran_data, X_long_data, X_w_data = X_tran_data.reshape(X_tran_data.shape[0], 1), X_long_data.reshape(X_long_data.shape[0], 1), X_w_data.reshape(X_w_data.shape[0], 1)
    Y_long_data, Y_w_data = Y_long_data.reshape(Y_long_data.shape[0], 1), Y_w_data.reshape(Y_w_data.shape[0], 1)
    X_data = np.concatenate((X_long_data, X_w_data), axis = 1)
    Y_data = np.concatenate((Y_long_data, Y_w_data), axis = 1)

    # use past information to give an approximation about Vx and Vy
    Vx, Vy = Vx.reshape(Vx.shape[0], 1), Vy.reshape(Vy.shape[0], 1)
    pre_Vx, pre_Vy = np.matmul(X_data, pre_VW), np.matmul(Y_data, pre_VW)
    prefactors = np.abs((Vx - pre_Vx) / pre_Vx ) + np.abs((Vy - pre_Vy) / pre_Vy)
    filter_mask = prefactors <= discard_threshold
    filter_mask = filter_mask.reshape(filter_mask.shape[0])

    return good_old[filter_mask], flow[filter_mask]

def f_extreme(op_Vl, op_w, Errors):
    """function be called when the regression failed, this typically happened in extreme conditions, for example, the ground is completely smooth and there is no feature to track at all, or the environment is completely dark
    key idea: again, apply past information as a backup source for information at this emergency situation"""

    print("""Egomotion Estimation Regression FAILED !!!!!!!!!!!! f_extreme called""")
    past_steps = 10
    past_len = len(op_Vl)
    if past_len == 0:
        print("f_extreme called when no past information available")
        return 0, 0, 0, 0, 0
    past_steps = min(past_len, past_steps)

    past_amplifier = 1.2
    past_Vls, past_Ws, past_Errors = op_Vl[- past_steps : ], op_w[-past_steps : ], Errors[- past_steps : ]

    # now take the average of past measurements and past errors
    past_Vl, past_w, past_Error = np.average(past_Vls), np.average(past_Ws), np.average(past_Errors)
    past_Error *= past_amplifier

    return past_Vl, past_w, past_Error, past_Vl, past_w

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