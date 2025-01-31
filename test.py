import numpy as np
import data_utils as du
import OP_utils as pu
import Ego_motion as em
import matplotlib.pyplot as plt
import video_utils as vu
import Estimators

# the summationon of names of packages used
BARC_PATH = "ParaDriveLocalComparison_Oct1_enc_0.npz"
CARLA_PATH1 = "carlaData1.pkl"
CARLA_PATH2 = "carlaData2.pkl"
CHESS_STRAIGHT = "carlaData3.pkl"
CHESS_STRAIGHT2 = "chessStraight2.pkl"
CHESS_STRAIGHT3 = "chessStraight3.pkl"
CHESS_CIRCLE = "ChessCircle.pkl"
LADDER1 = "ladder1.pkl"
LADDER2 = "ladder2.pkl"
REAL1 = "real1.pkl"


"""to replicate the experiment, download the data packages from the google drive
the following packages are recommended to be used to see how the diversity of pixel intensity affects the performance of optical-flow measurement

CHESS_STRAIGHT3: the packages collected in the most ideal road scenario in which the road is paved with chessboard texture, i.e. there are ample pixel points which have enough changes in the intensity of the gradients in its neighborhood. In such a scenario, optical flow works the best.
LADDER2: a relatively ideal road scenario data package, in which we can still see a great amount of good features to track.
REAL1: the packges collected in the most realistic road scenario in which road markings and signs appear randomly at a low frequency, i.e. most of the time, optical flow won't work quite well

We have three test functions
rgb_test: the test function that only tests the performance of optical flow measurement
imu_test: the test function that only uses the measurements of imu to estimate the ego-motion
full_test: a test function that fuses the estimations from both imu and optical measurements

for rgb_test and full_test, we have a paramenter, show_img:
    if show_img is true, the test function will draw the image at the correpsonding frame and stops there until a key is pressed to make the estimation for the next frame
    if show_img is false, the test function will show no image, it will keep making estimations and compare the real values and the estimations in the end"""

def estimator_test(Path = REAL1, start_frame = 10, end_frame = 500, show_img = False):
    images, real_Vl, real_w, f, h, deltaT = du.full_parse(Path)
    estimator = Estimators.OP_estimator(deltaT = deltaT, h = h, f = f, start_image=images[start_frame], pre_filter_discard=3.0, pre_filter_size=10, past_fusion_size = 20, past_fusion_amplifier = 1.2, past_fusion_on= False)
    est_Vls, est_ws, est_errors = [], [], []
    for i in range(start_frame + 1, end_frame):
        nextImg = images[i]
        est_Vl, est_w, cur_error = estimator.estimate(nextImg)
        est_Vls.append(est_Vl)
        est_ws.append(est_w)
        est_errors.append(cur_error)

    est_Vls = em.median_filter(est_Vls)
    est_ws = em.median_filter(est_ws)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(real_Vl[start_frame : end_frame], label = "real_Vl")
    ax1.plot(est_Vls, label = "est_vl")
    ax1.set_xlabel("frame number")
    ax1.set_ylabel("V_long (m / s)")
    ax2.plot(real_w[start_frame : end_frame], label = "real_w")
    ax2.plot(est_ws, label = "est_w")
    ax2.set_xlabel("frame number")
    ax2.set_ylabel("w (rad / s)")
    ax3.plot(est_errors, label = "error estimated")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

def rgb_test(Path = CHESS_STRAIGHT3, start_frame = 10, end_frame = 500, show_img = False):
    """this test only involves the perception of optical flow optical no measurement from imu is included"""
    images, real_Vl, real_w, f, h, deltaT = du.full_parse(Path)
    op_Vt, op_Vl, op_w = [], [], [] # all values estiamted by pure optical flow
    est_Vt, est_Vl, est_w = [], [], [] # values that are optimized by Kalman filter and other algorithms
    Errors = [] # record the errors of past estimations

    for i in range(start_frame, end_frame):
        preImg, nextImg = images[i], images[i + 1]
        mask = pu.G_cutoff(preImg)
        good_old, good_new, flow = pu.cv_featureLK(preImg, nextImg, deltaT, mask)

        print(f"""---- Inspecting the size of the data points of frame {i}--------
Before the prefilter: {good_old.shape[0]}""")
        
        good_old, good_new, flow = em.pre_filter(good_old, good_new, flow, h, f, op_Vl, op_w)

        print(f"after pre_filter: {good_old.shape[0]}")

        try:
          V_long, w, Error = em.WVReg(good_old, flow, h, f) # use egomotion estimation function to estimate the ego motion
          # apply optimization algorithms here
          final_V_long = em.past_fusion(op_Vl, Errors, V_long, Error)
          final_w = em.past_fusion(op_w, Errors, w, Error)

          # sometimes there might be some extreme conditions that make regression failed, i.e. almost no flow point can be used for regression at this time an index error will be raised in WVReg
        except IndexError:
            V_long, w, Error = em.past_extreme(op_Vl, op_w, Errors)
            final_V_long, final_w = V_long, w
        if show_img:
            vu.drawFlow(preImg, good_old, good_new)


        op_Vl.append(V_long)
        op_w.append(w)
        est_Vl.append(final_V_long)
        est_w.append(final_w)
        Errors.append(Error)

        # have some printing imformation every 7 frames
        if i % 7 == 0:
            print(f"""///////////////////// at frame {i} ////////////////////
                  op_V_long = {V_long}, op_w = {w}, 
                  est_V_long = {final_V_long}, est_w = {final_w},
                  real_V_long = {real_Vl[i]}, real_w = {real_w[i]}""")
    
    print(f"the average noise that existed in optical measurement of vl is {em.noise_est(real_Vl[start_frame : end_frame], op_Vl)}")
    print(f"the average noise that existed in optical measurement of w is {em.noise_est(real_w[start_frame : end_frame], op_w)}")

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.plot(real_Vt[start_frame : end_frame], label = "real_V_tran")
    # ax1.plot(op_Vt, label = "op_V_tran")
    ax2.plot(real_Vl[start_frame : end_frame], label = "real_V_long")
    ax2.set_xlabel("frame number")
    ax2.set_ylabel("V_long (m / s)")
    # ax2.plot(op_Vl, label = "op_V_long")
    ax2.plot(est_Vl, label = "est_V_long")
    ax3.plot(real_w[start_frame : end_frame], label = "real_w")
    # ax3.plot(op_w, label = "op_w")
    ax3.plot(est_w, label = "est_w")
    ax3.set_xlabel("frame number")
    ax3.set_ylabel("w (rad / s)")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

def imu_test(start_frame = 10, end_frame = 200, Path = REAL1):
    """this function only tests the performance of imu sensor with regarding to the real sesnor"""
    images, real_Vl, real_w, f, h, deltaT = du.full_parse(Path)
    imu_data = du.imu_parse(Path)

    Vl0 = real_Vl[start_frame]

    imu_w = imu_data[start_frame : end_frame, 1]
    imu_al = imu_data[start_frame : end_frame, 0]

    imu_vl = em.f_as2vls(imu_al, deltaT, Vl0)[start_frame : end_frame]
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(real_Vl[start_frame : end_frame], label = "real_Vl")
    ax1.plot(imu_vl, label = "imu_vl")
    ax1.set_xlabel("frame number")
    ax1.set_ylabel("V_long (m / s)")
    ax2.plot(real_w[start_frame : end_frame], label = "real_w")
    ax2.plot(imu_w, label = "imu_w")
    ax2.set_xlabel("frame number")
    ax2.set_ylabel("w (rad / s)")
    print(f"the average noise that existed in imu measurement of w is {em.noise_est(real_w[start_frame : end_frame], imu_w)}")
    ax1.legend()
    ax2.legend()
    plt.show()

def full_test(Path = REAL1, start_frame = 10, end_frame = 500, show_img = False):
    """test function for fused imu and rgb estimation"""
    images, real_Vl, real_w, f, h, deltaT = du.full_parse(Path)
    imu_data = du.imu_parse(Path)

    # the emperical measured value of noise
    imu_w_noise = 0.23
    rgb_w_noise = 0.14

    op_Vt, op_Vl, op_w = [], [], [] # all values estiamted by pure optical flow
    est_Vt, est_Vl, est_w = [], [], [] # values that are optimized by Kalman filter and other algorithms
    vl_errors = [] # record the errors of past estimations

    for i in range(start_frame, end_frame):
        preImg, nextImg = images[i], images[i + 1]
        al_imu, w_imu = imu_data[i, 0], imu_data[i , 1]

        mask = pu.G_cutoff(preImg)
        good_old, good_new, flow = pu.cv_featureLK(preImg, nextImg, deltaT, mask)

        try:
          vl_imu, imu_vl_noise = em.f_a2vl(al_imu, deltaT, est_Vl)
          good_old, good_new, flow = em.pre_filter(good_old, good_new, flow, h, f, op_Vl, op_w)
          vl_rgb, w_rgb, error = em.WVReg(good_old, flow, h, f) # use egomotion estimation function to estimate the ego motion
          rgb_vl_noise = em.f_vlNoise(error)

          vl_final = em.Kalman_filter(vl_imu, vl_rgb, imu_vl_noise, rgb_vl_noise, vl_errors)
          w_final = em.simple_fusion(w_rgb, w_imu, rgb_w_noise, imu_w_noise)
          # sometimes there might be some extreme conditions that make regression failed, i.e. almost no flow point can be used for regression at this time an index error will be raised in WVReg

        except IndexError:
            vl_rgb, w_rgb, vl_final, w_final = em.imu_extreme(vl_imu, w_imu, vl_errors, imu_vl_noise)
        
        if show_img:
            print(f"""///////////////////// at frame {i} ////////////////////
                  op_V_long = {vl_rgb}, op_w = {w_rgb}, 
                  est_V_long = {vl_final}, est_w = {w_final},
                  real_V_long = {real_Vl[i]}, real_w = {real_w[i]}
                  the average residuals of rgb is {error}""")
            vu.drawFlow(preImg, good_old, good_new)

        op_Vl.append(vl_rgb)
        op_w.append(w_rgb)
        est_Vl.append(vl_final)
        est_w.append(w_final)

        # have some printing imformation every 7 frames
        if not show_img:
            if i % 7 == 0:
                print(f"""///////////////////// at frame {i} ////////////////////
                  op_V_long = {vl_rgb}, op_w = {w_rgb}, 
                  est_V_long = {vl_final}, est_w = {w_final},
                  real_V_long = {real_Vl[i]}, real_w = {real_w[i]}""")
                
    est_w = em.median_filter(est_w)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.plot(real_Vt[start_frame : end_frame], label = "real_V_tran")
    # ax1.plot(op_Vt, label = "op_V_tran")
    ax2.plot(real_Vl[start_frame : end_frame], label = "real_V_long")
    # ax2.plot(op_Vl, label = "op_V_long")
    ax2.plot(est_Vl, label = "est_V_long")
    ax2.set_xlabel("frame number")
    ax2.set_ylabel("V_long (m / s)")
    ax3.plot(real_w[start_frame : end_frame], label = "real_w")
    # ax3.plot(op_w, label = "op_w")
    ax3.plot(est_w, label = "est_w")
    ax3.set_xlabel("frame number")
    ax3.set_ylabel("w (rad / s)")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

def main():
    estimator_test()

if __name__ == "__main__":
    main()
