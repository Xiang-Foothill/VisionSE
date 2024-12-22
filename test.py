import numpy as np
import data_utils as du
import OP_utils as pu
import Ego_motion as em
import matplotlib.pyplot as plt

def rgb_test():
    """this test only involves the perception of optical flow optical no measurement from imu is included"""
    images, real_Vl, real_w, f, h, deltaT = du.full_parse()
    op_Vt, op_Vl, op_w = [], [], [] # all values estiamted by pure optical flow
    est_Vt, est_Vl, est_w = [], [], [] # values that are optimized by Kalman filter and other algorithms
    Errors = [] # record the errors of past estimations
    start_frame = 10
    end_frame = 200

    for i in range(start_frame, end_frame):
        preImg, nextImg = images[i], images[i + 1]
        mask = pu.G_cutoff(preImg)
        good_old, good_new, flow = pu.cv_featureLK(preImg, nextImg, deltaT, mask)

        print(f"""---- Inspecting the size of the data points of frame {i}--------
Before the prefilter: {good_old.shape[0]}""")
        
        good_old, flow = em.pre_filter(good_old, flow, h, f, op_Vl, op_w)

        print(f"after pre_filter: {good_old.shape[0]}")

        try:
          V_long, w, Error = em.WVReg(good_old, flow, h, f) # use egomotion estimation function to estimate the ego motion
          # apply optimization algorithms here
          final_V_long = em.Kalman_filter(op_Vl, Errors, V_long, Error)
          final_w = em.Kalman_filter(op_w, Errors, w, Error)

          # sometimes there might be some extreme conditions that make regression failed, i.e. almost no flow point can be used for regression at this time an index error will be raised in WVReg
        except IndexError:
            V_long, w, Error, final_V_long, final_w = em.f_extreme(op_Vl, op_w, Errors)

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

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.plot(real_Vt[start_frame : end_frame], label = "real_V_tran")
    # ax1.plot(op_Vt, label = "op_V_tran")
    ax2.plot(real_Vl[start_frame : end_frame], label = "real_V_long")
    # ax2.plot(op_Vl, label = "op_V_long")
    ax2.plot(est_Vl, label = "est_V_long")
    ax3.plot(real_w[start_frame : end_frame], label = "real_w")
    # ax3.plot(op_w, label = "op_w")
    ax3.plot(est_w, label = "est_w")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

def imu_test():
    """this function only tests the performance of imu sensor with regarding to the real sesnor"""
    images, real_Vl, real_w, f, h, deltaT = du.full_parse()
    imu_data = du.imu_parse()

    start_frame = 10
    Vl0 = real_Vl[0]
    end_frame = 400

    imu_w = imu_data[start_frame : end_frame, 1]
    imu_al = imu_data[start_frame : end_frame, 0]

    imu_vl = em.f_a2vl(imu_al, deltaT, Vl0)[start_frame : end_frame]
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(real_Vl[start_frame : end_frame], label = "real_Vl")
    ax1.plot(imu_vl, label = "imu_vl")
    ax2.plot(real_w[start_frame : end_frame], label = "real_w")
    ax2.plot(imu_w, label = "imu_w")
    ax1.legend()
    ax2.legend()
    plt.show()

def main():
    rgb_test()

if __name__ == "__main__":
    main()