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

def get_Trackpoints(img, ground_mask):
    # specify the function used to detect the ground
    # f_ground = G_cutoff

    # find the points with good features to be tracked
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
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

"""********** MASK function below ***************"""

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

def side_ground(img: np.array) -> np.array:
    """generate a mask in which only the left 1/3 sides and right 1/3 sides of the ground are marked as True"""

    H, W = img.shape[0], img.shape[1]
    # define the height and width of the ground zone
    ground_H = H * 0.38
    ground_W = W / 4
    ground_start = 0
    bottom_center1 = W / 6
    bottom_center2 = W * 5 / 6

    mask = np.zeros(shape = (H, W), dtype = bool)
    mask[int(H - ground_H - ground_start): int(H - ground_start), int(bottom_center1 - ground_W / 2): int(bottom_center1 + ground_W / 2)] = True
    mask[int(H - ground_H - ground_start): int(H - ground_start), int(bottom_center2 - ground_W / 2) : int(bottom_center2 + ground_W / 2)] = True
    return mask

def bottom_ground(img):
    """generate a mask in which only the bottom 1/2 of the ground is marked as true"""

    H, W = img.shape[0], img.shape[1]
    # define the height and width of the ground zone
    ground_H = H * 0.30
    ground_W = W - 10
    ground_start = 0
    bottom_center = W * 0.5

    mask = np.zeros(shape = (H, W), dtype = bool)
    mask[int(H - ground_H - ground_start): int(H - ground_start), int(bottom_center - ground_W / 2): int(bottom_center + ground_W / 2)] = True
    return mask

def X0_ground(img):
    """generate a mask in which all the points close to the vertical axis x = 0 are marked as true"""

    H, W = img.shape[0], img.shape[1]
    # define the height and width of the ground zone
    ground_H = H * 0.25
    ground_W = 60
    ground_start = 0
    bottom_center = W * 0.5

    mask = np.zeros(shape = (H, W), dtype = bool)
    mask[int(H - ground_H - ground_start): int(H - ground_start), int(bottom_center - ground_W / 2): int(bottom_center + ground_W / 2)] = True
    return mask

def regionGround(img, region_name):
    """the region mask function in the general form
    region_name = "topLeft" / "topMiddle" ect. 
    based on the name of the region the correspondig region is marked as true"""

    # the helper function
    def mark_region(y_start, y_end, x_start, x_end):

        H, W = img.shape[0], img.shape[1]
        mask = np.zeros(shape = (H, W), dtype = bool)
        mask[y_start: y_end, x_start : x_end] = True

        return mask
    
    H, W = img.shape[0], img.shape[1]

    # define the boundaries for the region
    h0, h1, h2 = int(H * 0.5), int(H * 0.75), H
    w0, w1, w2, w3 = 0, int(W / 3), int(W * 2 / 3), W

    if region_name == "topLeft":
        return mark_region(h0, h1, w0, w1)
    if region_name == "topMiddle":
        return mark_region(h0, h1, w1, w2)
    if region_name == "topRight":
        return mark_region(h0, h1, w2, w3)
    if region_name == "downLeft":
        return mark_region(h1, h2, w0, w1)
    if region_name == "downMiddle":
        return mark_region(h1, h2, w1, w2)
    if region_name == "downRight":
        return mark_region(h1, h2, w2, w3)
    
# coord: the coordinates of the pixels to be changed
def recenter(coord):
    """relocate the origin of the pixel system to the center of the image plane
    the original pixel system takes the top-left corner as (0, 0)"""
    new_coord = coord.copy()
    W = 640
    H = 480
    # new_coord[:, 0] = np.abs(H * 0.5 - new_coord[:, 0])
    # new_coord[:, 1] = np.abs(new_coord[:, 1] - W * 0.5)

    new_coord[:, 0] = H * 0.5 - new_coord[:, 0]
    new_coord[:, 1] = new_coord[:, 1] - W * 0.5
    return new_coord

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

def centerBack(coord):
    """relocate the origin of the pixel system back to the top left"""
    new_coord = coord.copy()
    W = 640
    H = 480

    new_coord[:, 0] = new_coord[:, 0] + W * 0.5 # change the y-coordinates
    new_coord[:, 1] = new_coord[:, 1] + H * 0.5 # change the x-coordinates

    return new_coord

def f_str(V):
    """saturation function that prevents the speed estimation from overshooting"""
    #check if the observed velocity is abnormal, if the velocity measured is abnormal, set it to 1
    abnormal_threshold = 10
    default_V = 1
    if abs(V) > abnormal_threshold:
        V = default_V
    
    return V

def VxReg(good_old, flow, h, f, plot_tool):
    """apply linear regression to find the most optimized estimation of egoVehicle speed
    Vx = (x*y) * (V / hf)
    treat V / hf as the gradient of the line and Vx as the Y value and (x * y) as the X value"""
    # decide to plot the point ditribution or not
    index = plot_tool["index"]
    N = plot_tool["N"]
    axs = plot_tool["axs"]
    row = plot_tool["row"]
    plot_prob = 0.01
    plot_bool = np.random.uniform(low = 0.0, high = 1.0) < plot_prob
    plot_bool = plot_prob and index < N

    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    a = x * y
    b = Vx
    # b = np.abs(b)
    a = a.reshape((a.shape[0], 1))
    res = np.linalg.lstsq(a, b)
    V = res[0][0] * h * f
    Error = res[1][0] / 3000000

    if plot_bool:
        real_grad = plot_tool["realV"] / (h * f)
        aReal = np.linspace(start = 0, stop = 100000)
        bReal = aReal * real_grad
        bEst = aReal * res[0][0]
        axs[index // row][index % row].scatter(a, b, s = 5)
        axs[index // row][index % row].plot(aReal, bReal, label = "realV")
        axs[index // row][index % row].plot(aReal, bEst, label = "opV")
        axs[index // row][index % row].set_title("Vx vs x * y")
        axs[index // row][index % row].legend()
        plot_tool["index"] = index + 1
    return V, Error

def VyReg(good_old, flow, h, f, plot_tool):
    """apply linear regression to find the most optimized estimation of egoVehicle speed
    Vy = (y ^ 2) * (V / hf)
    treat V / hf as the gradient of the line and Vy as the Y value and (y ^ 2) as the X value"""

    # decide to plot the point ditribution or not
    index = plot_tool["index"]
    N = plot_tool["N"]
    axs = plot_tool["axs"]
    row = plot_tool["row"]
    plot_prob = 0.01
    plot_bool = np.random.uniform(low = 0.0, high = 1.0) < plot_prob
    plot_bool = plot_prob and index < N

    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    a = y * y
    b = Vy
    # b = np.abs(b)
    a = a.reshape((a.shape[0], 1))
    res = np.linalg.lstsq(a, b)
    V = res[0][0] * h * f
    Error = res[1][0] / 3000000

    if plot_bool:
        real_grad = plot_tool["realV"] / (h * f)
        aReal = np.linspace(start = 0, stop = 100000)
        bReal = aReal * real_grad
        bEst = aReal * res[0][0]
        axs[index // row][index % row].scatter(a, b, s = 5)
        axs[index // row][index % row].plot(aReal, bReal, label = "realV")
        axs[index // row][index % row].plot(aReal, bEst, label = "opV")
        axs[index // row][index % row].set_title("Vy vs y^2")
        axs[index // row][index % row].legend()
        plot_tool["index"] = index + 1
    return V, Error

# f_op: the optical flow function to be used
# deltaT: the time constant between two consecutive frames
# This method has poor performance for speed estimation
def simpleVego(good_old, flow, h, f):
    """recover the egomotion from the image velocity
    Use the simplified version of plane-motion-filed equation (emega is ignored): 
    Vx = (v * x * y) / (h * f)
    Vy = (v * y ^ 2) / (h * f)
    
    find the egomotion velocity by averaging the calculated velocities of all the output flow points given by the f_op function"""
    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    v1 = np.abs((Vx * h * f) / (x * y))
    v2 = np.abs((Vy * h * f) / (y * y))
    return np.concatenate((v1, v2))

def Vx2V(good_old, flow, h, f):
    """this function is the same as simpleVego
    However, it only applies the value of Vx to evaluate the speed of the ego vehicle"""

    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    v1 = np.abs((Vx * h * f) / (x * y))

    return v1

def Vy2V(good_old, flow, h, f):
    """this function is the same as simpleVego
    However, it only applies the value of Vy to evaluate the speed of the ego vehicle"""

    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    v2 = np.abs((Vy * h * f) / (y * y))
    return v2

def Vx2W(good_old, flow, h, f):
    """find the value of angular velocities based on purely Vx
    all the points in good_old are assumed to have x_coordinates with very small values"""
    Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
    Ws = Vx / f
    return - Ws

def preFilter(Xs, preX, discard_threshold):
    """discard the abnormal outliers from Vs"""

   # discard speed V if V > preV + discard_t or V < preV - discard_t
    high, low = preX + discard_threshold, preX - discard_threshold
    keep_mask = np.logical_and(Xs < high, Xs > low)
    Xs = Xs[keep_mask]

    return Xs

def fullEq(good_old, flow, h, f):
    """apply the full version of the egomotion equation including angular velocity
    Vx = (x^2 + f^2 ) * a + xyb
    Vy = xya + y^ 2 * b
    where a = omega / f
    b = v / hf"""
    good_old = recenter(good_old)
    Vxs, Vys, xs, ys = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]

    #TO DO: replace the following codes with numy broadcasting operations to make it faster
    N = len(Vxs)
    omegas = []
    Vs = []

    for point_index in range(N):
        Vx, Vy, x, y = Vxs[point_index], Vys[point_index], xs[point_index], ys[point_index]
        A = np.asarray([[x ** 2 + f ** 2, x * y], [x * y, y ** 2]])
        goals = [[Vx], [Vy]]
        try:
            A_inverse = np.linalg.inv(A)
        except np.linalg.LinAlgError as err:
            continue
        res = np.matmul(A_inverse, goals)
        a, b = res[0, 0], res[1, 0]
        omegas.append(a * f), Vs.append(b * h * f)
    
    omegas = np.asarray(omegas)
    Vs = np.asarray(Vs)
    return np.abs(Vs), np.abs(omegas)

def Dpre(Vs, Vpre):
    """evaluate the Dpre factor which determines how close the speed estimations are to the previous speed
    the closer the better"""
    D_weight = 1 # the full credit a point can receive
    thetas = (Vs - Vpre) / Vpre # the distance as a fraction of Vpre
    minD = 0.2 # if theta is smaller than or equal to 0.2 it will receive a full credit for Dpre

    for i, theta in enumerate(thetas):
        thetas[i] = max(minD, theta)

    def make_D_f():
        point1, credit1 = 0.8, 0.3
        point2, credit2 = 1.0, 0.1
        A = np.asarray([[minD**2, minD, 1], 
                        [point1 ** 2, point1, 1],
                        [point2 ** 2, point2, 1]])
        b = np.asarray([[D_weight], [credit1], [credit2]])
        parameters = np.matmul(np.linalg.inv(A), b)
        return parameters[:, 0]
    
    [par1, par2, par3] = make_D_f()
    credits = par1 * thetas ** 2 + par2 * thetas + par3
    credits = np.clip(credits, 0, 1)
    return credits

# Vs: the new estimations of velocities given by the optical flow of pixels
# preV: the velocity at t - 1 step
# preE: the error of estimation of the previous step
def simpleKalman(Xs, preX, preE, X_noise, returnStd = False):
    """implement the simple one-variable version of kalman filter
    1. regard all the pixels as one single sensor
    2. the error provided by the measurement of this single sensor is equal to the standard deviation of Vs
    3. kalman_gain = Eest / (Eest + Emea)"""

    #note that the velocity is always changing, which means that using preV as an estimation for the current velocity will always bring in some noise
    preE += X_noise

    # find the measurement value and the error in the measurement
    X_mea, E_mea = np.average(Xs), np.std(Xs)
    print(f"the speed estimated by pure optical flow is {X_mea}")
    # E_mea = E_mea / V_mea

    Kalman_gain = preE / (preE + E_mea)
    X = preX + Kalman_gain * (X_mea - preX)
    newE = (1 - Kalman_gain) * preE

    if not returnStd:
        return X, newE
    if returnStd:
        return X, newE, E_mea

# preXs: the history records of the X values
#windowSize: the size of the sliding window with which we are going to perform Kalman filter
#newX: the value of newly measured X
def SWKalman(preXs, windowSize, newX, newE = 2.0):
    """implement kalman filter with the idea of sliding window:
    use the window to take samples from past measured values of X (assume the errors are the samples standard deviation)
    use these samples to combine with the newly measured X"""
    sampleX = preXs[- windowSize : ] # take samples from the nearest window-sized samples
    preX, preE = np.average(sampleX), np.std(sampleX)
    Kalman_gain = preE / (preE + newE)
    X = preX + Kalman_gain * (newX - preX)

    return X

"""####################### TEST FUNCTIONS BELOW ###########################"""

def V_test():
    images, real_V, f, h, deltaT = du.parse_barc_data()
    op_V = []
    sample_size = 100

    for i in range(0, sample_size):
        pre_img, next_img = images[i], images[i + 1]
        good_old, good_new, flow = cv_featureLK(pre_img, next_img, deltaT)
        V = simpleVego(good_old, flow, h, f)
        V = f_str(V)
        op_V.append(V)
    real_V = real_V[: sample_size]
    plt.plot(real_V, label = "real_V")
    plt.plot(op_V, label = "op_V")
    plt.legend()
    plt.show()

def abnormal_test():
    images, real_V, f, h, deltaT = du.parse_barc_data()
    op_V = []
    start_frame = 0
    sample_size = 200

    for i in range(start_frame, start_frame + sample_size):
        pre_img, next_img = images[i], images[i + 1]
        good_old, good_new, flow = cv_featureLK(pre_img, next_img, deltaT)
        V = simpleVego(good_old, flow, h, f)
        V = f_str(V)
        print(f"the estimated speed at frame {i} is {V}, the real speed is {real_V[i]}")
        print(pre_img.shape)
        vu.drawFlow(pre_img, good_old, good_new)

def selectedTest(show_img, mode):
    images, real_V, real_Omega, f, h, deltaT = du.parse_barc_data(Omega_exist=True)
    op_V = []
    OP_Omega = []

    start_frame = 80
    sample_size = 100
    preV = real_V[start_frame]
    preW = real_Omega[start_frame]
    V_discard = 20
    W_discard = 5.0
    V_noise = 0.05
    W_noise = 0.01
    preVE = 0.0
    preWE = 0.0

    for i in range(start_frame, start_frame + sample_size):
        pre_img, next_img = images[i], images[i + 1]

        if mode == "onlyV":
            mask1, mask2 = side_ground(pre_img), bottom_ground(pre_img)
            good_old1, good_new1, flow1 = cv_featureLK(pre_img, next_img, deltaT, mask1)
            good_old2, good_new2, flow2 = cv_featureLK(pre_img, next_img, deltaT, mask2)
            flow = np.concatenate((flow1, flow2))
            good_old = np.concatenate((good_old1, good_old2))
            good_new = np.concatenate((good_new1, good_new2))
            Vs1, Vs2 = Vx2V(good_old1, flow1, h, f), Vy2V(good_old2, flow2, h, f)
            Vs = np.concatenate((Vs2, ))
            Vs = preFilter(Vs, preV, V_discard)
            V, preVE = simpleKalman(Vs, preV, preVE, V_noise)
            preV = V
            W, preW = 0.0, 0.0
        
        elif mode == "fullEq":
            mask = G_cutoff(pre_img)
            good_old, good_new, flow = cv_featureLK(pre_img, next_img, deltaT, mask)
            Vs, Omegas = fullEq(good_old, flow, h, f)
            Vs, Omegas = preFilter(Vs, preV, V_discard), preFilter(Omegas, preW, W_discard)
            V, preVE = simpleKalman(Vs, preV, preVE, V_noise)
            W, preWE = simpleKalman(Omegas, preW, preWE, W_noise)
            preV, preW = V, W
        
        elif mode == "onlyW":
            mask = X0_ground(pre_img)
            good_old, good_new, flow = cv_featureLK(pre_img, next_img, deltaT, mask)
            Ws = Vx2W(good_old, flow, h, f)
            Ws = preFilter(Ws, preW, W_discard)
            W, preWE = simpleKalman(Ws, preW, preWE, W_noise)
            preW = W
            V, preV = 0.0 , 0.0

        op_V.append(V), OP_Omega.append(W)
        print(f"At the frame {i}: V_estimated = {V} real_V = {real_V[i]}; W_estimated = {W}, real_W = {real_Omega[i]}")
        if show_img:
            Vx, Vy, x, y = flow[:, 0], flow[:, 1], good_old[:, 0], good_old[:, 1]
            real_grad = real_V[i] / (h * f)
            aReal = np.linspace(start = 0, stop = 100000)
            bReal = aReal * real_grad
            a = y* y
            b = Vy
            plt.scatter(a, b, s = 5)
            plt.plot(aReal, bReal, label = "realV")
            plt.show()
            vu.drawFlow(pre_img, good_old, good_new)
    
    # plt.plot(real_V[start_frame:start_frame + sample_size], label = "real_V")
    # plt.plot(op_V, label = "op_V")
    # plt.legend()
    # plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(real_V[start_frame : start_frame + sample_size], label = "real_V")
    ax1.plot(op_V, label = "op_V")
    ax2.plot(real_Omega[start_frame : start_frame + sample_size], label = "real_Omega")
    ax2.plot(OP_Omega, label = "op_Omega")
    ax1.legend()
    ax2.legend()
    plt.show()


# test the performance of each region on evaluating speed and angular speed
def regionTest(mode, region_displayed):
    images, real_V, real_Omega, f, h, deltaT = du.parse_barc_data(Omega_exist=True)
    OP_Omega = []

    start_frame = 60
    sample_size = 100
    preV = real_V[start_frame]
    preW = real_Omega[start_frame]
    V_discard = 20
    W_discard = 5.0
    V_noise = 0.05
    W_noise = 0.01
    preVE = 0.0
    preWE = 0.0
    regions = np.asarray([["topLeft", "topMiddle", "topRight"], ["downLeft", "downMiddle", "downRight"]])

    def Vx_iter(mask):
        op_V = []
        preV = real_V[start_frame]
        preVE = 0.0
        display_set = {}
        for i in range(start_frame, start_frame + sample_size):
            pre_img, next_img = images[i], images[i + 1]

            good_old, good_new, flow = cv_featureLK(pre_img, next_img, deltaT, mask)
            Vs = Vx2V(good_old, flow, h, f)
            Vs = preFilter(Vs, preV, V_discard)
            V, preVE = simpleKalman(Vs, preV, preVE, V_noise)
            preV = V
            op_V.append(V)

            if np.random.uniform(low = 0.0, high = 1.0) > 0.95:
                display_set[f'frame {i}'] = vu.npArrowFlow(pre_img, good_old, good_new)

        return op_V, display_set
    
    def Vy_iter(mask):
        op_V = []
        preV = real_V[start_frame]
        preVE = 0.0
        display_set = {}
        for i in range(start_frame, start_frame + sample_size):
            pre_img, next_img = images[i], images[i + 1]

            good_old, good_new, flow = cv_featureLK(pre_img, next_img, deltaT, mask)
            Vs = Vy2V(good_old, flow, h, f)
            Vs = preFilter(Vs, preV, V_discard)
            V, preVE = simpleKalman(Vs, preV, preVE, V_noise)
            preV = V
            op_V.append(V)

            if np.random.uniform(low = 0.0, high = 1.0) > 0.95:
                display_set[f'frame {i}'] = vu.npArrowFlow(pre_img, good_old, good_new)
            
        return op_V, display_set
    
    fig, axs = plt.subplots(2, 3)

    for index, name in np.ndenumerate(regions):
        cur_region = name
        mask = regionGround(images[start_frame], region_name= cur_region)
        if mode == "Vx":
            iter = Vx_iter
        if mode == "Vy":
            iter = Vy_iter
        op_V, cur_list = iter(mask)
        axs[index].plot(op_V, label = "op_V")
        axs[index].plot(real_V[start_frame : start_frame + sample_size], label = "real_V")
        axs[index].set_title(cur_region)
        axs[index].legend()

    plt.show()

# this function is not workable for the pipeline rightnow
def full_estimator(preImg, nextImg, deltaT, h, f, preV, V_discard, preW, W_discard, preVE, preWE, V_noise, W_noise, history, mode, plot_tool, with_std = False):
    if mode == "onlyV":
        mask = regionGround(preImg, region_name = "topRight")
        # mask = G_cutoff(preImg)
        good_old, good_new, flow = cv_featureLK(preImg, nextImg, deltaT, mask)
        V_yest, E_yest = VyReg(good_old, flow, h, f, plot_tool = plot_tool)
        # V_xest, E_xest = VxReg(good_old, flow, h, f, plot_tool = plot_tool)
        V_xest, E_xest = V_yest, 0
        V = (V_yest + V_xest) / 2 # take the average of the speed estimated from the x_direction and the y_direction
        Error = (E_yest + E_xest) / 2
        filtered_V = SWKalman(history, windowSize = min(len(history), 10), newX = V, newE = Error)

        # Vs = simpleVego(good_old, flow, h, f)
        # mask1, mask2 = side_ground(preImg), bottom_ground(preImg)
        # good_old1, good_new1, flow1 = cv_featureLK(preImg, nextImg, deltaT, mask1)
        # good_old2, good_new2, flow2 = cv_featureLK(preImg, nextImg, deltaT, mask2)
        # good_old = np.concatenate((good_old1, good_old2))
        # good_new = np.concatenate((good_new1, good_new2))
        # Vs1, Vs2 = Vx2V(good_old1, flow1, h, f), Vy2V(good_old2, flow2, h, f)
        # Vs = np.concatenate((Vs1, ))

        # Vs = preFilter(Vs, preV, V_discard)
        # V, preVE, V_std = simpleKalman(Vs, preV, preVE, V_noise, returnStd = True)
        
        preV = V
        W, preW = 0.0, 0.0
    elif mode == "fullEq":
        good_old, good_new, flow = cv_featureLK(preImg, nextImg, deltaT)
        Vs, Omegas = fullEq(good_old, flow, h, f)
        Vs, Omegas = preFilter(Vs, preV, V_discard), preFilter(Omegas, preW, W_discard)
        V, preVE = simpleKalman(Vs, preV, preVE, V_noise)
        W, preWE = simpleKalman(Omegas, preW, preWE, W_noise)
        preV, preW = V, W

    if not with_std:
        return V, W, preV, preW, preVE, preWE, filtered_V
    if with_std:
        return V, W, preV, preW, preVE, preWE, filtered_V, V_std

# test the ground detection function
# draw_arrow == True when want to test it by drawing flows
# draw_arrow == False when want to test it by blacking out the ground area
def test_ground_mask(draw_arrow = True):
    images, real_V, f, h, deltaT = du.parse_barc_data()
    index = np.random.randint(low = 0, high = images.shape[0])
    preImg, nextImg = images[index], images[index + 1]
    if draw_arrow:
        good_old, good_new, flow = cv_featureLK(preImg, nextImg, deltaT)
        V = simpleVego(good_old, flow, h, f)
        vu.drawFlow(preImg, good_old, good_new)
    else:
        preImg = preImg.copy()
        mask = G_cutoff(preImg)
        img = cv2.cvtColor(preImg, cv2.COLOR_BGR2GRAY)
        img[mask] = 0
        cv2.imshow("image with the ground labeled", img)
        cv2.waitKey(0)

def main():
    # cv2.imwrite("result1.jpg", image)
    # test_ground_mask(False)
    # V_test()
    selectedTest(show_img = True, mode = "onlyV")
    # regionTest("Vy", region_displayed = ["downRight"])
    # test_ground_mask(draw_arrow = True)

if __name__ == "__main__":
    main()

# homography rectify