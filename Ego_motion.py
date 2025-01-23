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

    V_long_X, w_X, resid_X = res_X[0][0], - res_X[0][1], res_X[1][0] # we should have a negative sign for the angular velocity because our model takes downward as positive y-direction
    V_long_Y, w_Y, resid_Y = res_Y[0][0], - res_Y[0][1], res_Y[1][0]

    V_long, w, resid = (V_long_X + V_long_Y) / 2, (w_X + w_Y) / 2, (resid_X + resid_Y) / 2

    # measure the average deviation of the data points from the regression line, such deviation turns out to be a very effective measure to evaluate the quality of ego-motion estimation from optical flow
    error = ((resid / good_old.shape[0])) ** 0.5

    return  V_long, w, error

def noise_est(real_X, est_X):
    """an average noise estimator that measures the noise existed in est_X
    @real_X: the real value of X
    @est_X: the measured value of X"""
    dist = real_X - est_X
    return np.average(np.abs(dist))

def f_vlNoise(error):
    """a discrete function used to estimate the noise of the result given by WVReg
    @ error: the sauare root of average residuals of the regression result given by WVReg
    such error is very implicative about how good the estimation is
    Good estimation(low noise): error <= 20
    Moderate esimation(some noise): error <= 50
    bad estimation(high noise): error <= 500
    horrible estimation(extremeley noisy): error > 500"""
    low_noise, mid_noise, high_noise, extreme_noise = 0.1, 1.5, 5.0, 10.0
    if error <= 10.0:
        return low_noise
    elif error <= 20:
        return mid_noise
    elif error <= 100:
        return high_noise
    else:
        return extreme_noise

def simple_fusion(measure1, measure2, error1, error2):
    """a simple fusion function that combines measurements from two sensors without any iterated process"""
    gain = error1 / (error1 + error2)
    return (1 - gain) * measure1 + gain * measure2

def Kalman_filter(prediction, measurement, pred_noise, mea_error, error_history):
    """simple kalman filter that combines a 1-dimension prediction and a one dimension measurement
    @return:
    an estimation that combines the prediction value and the measurement value"""
    if not len(error_history):
        pre_error = 0
    else:
        pre_error = error_history[-1]
    
    pred_error = pred_noise + pre_error
    Kalman_gain = pred_error / (pred_error + mea_error)
    final = (1 - Kalman_gain) * prediction + Kalman_gain * measurement
    new_error = pred_error - Kalman_gain * pred_error

    error_history.append(new_error)

    return final

def past_fusion(op_Xs, op_errors, cur_X, error):
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

def pre_filter(good_old, good_new, flow, h, f, op_Vl, op_w):
    """key idea:
    use the information from past time (op_Vl, op_w) to filter out obvious outliers in the data, since the velocities of the car cannot change too much in very limited time
    for a point with Vx, Vy, and pre_Vx, pre_Vy
    its PREFACTOR = abs((Vx - pre_Vx) / pre_Vx ) + abs((Vy - pre_Vy) / pre_Vy)

    @ return: flow vector without obvious outliers"""
    discard_threshold = 5.0 # the threshold for discard a point, if a point's preFactor is higher than this threshold, discard it

    past_steps = 5  # the length of the horizon of how long we want to look back
    past_len = len(op_Vl)

    if past_len <= 10:
        return good_old, good_new, flow # if we are at the very start, this function is not applicable, return good_old, and flow directly
    
    # obtain past information
    past_Vl, past_w = np.median(op_Vl[- past_steps : ]), np.median(op_w[- past_steps : ])
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

    return good_old[filter_mask], good_new[filter_mask], flow[filter_mask]

def imu_filter(good_old, good_new, flow, h, f, vl_imu, w_imu):
    """key idea:
    similar to the pre_filter function above which relies on the history records to remove outliers from the regression data points,
    this function uses the available information of estimated vl and w given by imu to remove the outliers"""
    discard_threshold = 4.0
    imu_VW = np.asarray([[vl_imu], [w_imu]])
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
    pre_Vx, pre_Vy = np.matmul(X_data, imu_VW), np.matmul(Y_data, imu_VW)
    prefactors = np.abs((Vx - pre_Vx) / pre_Vx ) + np.abs((Vy - pre_Vy) / pre_Vy)
    filter_mask = prefactors <= discard_threshold
    filter_mask = filter_mask.reshape(filter_mask.shape[0])

    return good_old[filter_mask], good_new[filter_mask], flow[filter_mask]


def imu_extreme(vl_imu, w_imu, errors, vl_imu_noise):
    """a helper function that deals with the situation when the past regression process for optical flow fails"""
    print("Regression Failed!!!!! Extreme function called")
    errors.append(errors[-1] + vl_imu_noise)
    return vl_imu, w_imu, vl_imu, w_imu

def past_extreme(op_Vl, op_w, Errors):
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

    return past_Vl, past_w, past_Error

def f_as2vls(al, deltaT, Vl0):
    """apply accumulative operation to transform an acceleration time series into its correpsonding linear velocity time series
    @ a: a time series array with dimention (N, 2) [ax, xy]
    @ deltaT: the time interval between two adjacent index
    Vxy0: initial velocity in the x-direction and y-direction [Vx, Vy]"""
    al = median_filter(al)
    imu_vl = np.zeros(shape = (al.shape[0]))
    pre_vl = Vl0
    pre_al = 0

    for i in range(al.shape[0]):
        cur_vl = pre_vl + deltaT * pre_al
        imu_vl[i, ] = cur_vl
        pre_al = al[i, ]
        pre_vl = cur_vl
    
    print(f"the percentage error in acceleration measurement is {1.2 / np.median(al)}")
    return imu_vl

def f_a2vl(al, deltaT, est_vl):
    """predict the longitudinal velocity of the car at the next time step, given the linear acceleration estimated by imu, and the estimation of linear velocity at the last time step
    Different from the f_as2vls function above, this function only provides estimation at a single time step, instead of a whole time series
    input:
    est_vl: the history of estimated longitudinal velocity
    return:
    @ vl: the estimated longitudinal velocity at the next time step"""
    if not len(est_vl): # the case when we do not have any previous estimation of longitudinal speed, which makes prediction by linear acceleration impossible
        return 0, 10 ** 6
    vl0 = est_vl[-1]
    
    imu_vl_std = 0.12
    vl = vl0 + deltaT * al # estimation by linear acceleration model
    vl_noise = deltaT * imu_vl_std
    return vl, vl_noise

def median_filter(Xs, threshold = 50):
    """note that sometimes, the imu sensors will sometimes give some noisy measurements that are hundreds of times out of the normal range of meansurements, which is catastrophic for our accumulative measurements
    filter out these values in axy by applying median_value_filter
    @ threshold: how many times can the measured values exceed the median value, replace these obvious outliers with the average acceleration in the most recent ten steps"""
    x_median = np.median(Xs)
    replace_steps = 10
    for i in range(Xs.shape[0]):
        if abs(Xs[i]) > abs(x_median * threshold):
            x_replace = np.average(Xs[max(0, i - replace_steps) : i])
            Xs[i] = x_replace
    
    return Xs

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