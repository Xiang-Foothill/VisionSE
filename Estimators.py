import numpy as np
import data_utils as du
import OP_utils as pu
import Ego_motion as em
import matplotlib.pyplot as plt
import video_utils as vu

class OP_estimator:
    """estimator class for optical flow estimation"""

    def __init__(self, deltaT, h, f, start_image = None):

        self.f = f
        self.h = h
        self.deltaT = deltaT
        self.preImg = start_image

        # the recording arrays for the estimation
        self.raw_opW = []
        self.raw_opVl = []
        self.filter_opW = []
        self.filter_opVl = []
        self.raw_Errors = []
        self.filter_Errors = []
    
    def setPreImg(self, preImg):
        self.preImg = preImg

    def estimate(self, nextImg):
        if self.preImg == None:
            print("""Error: the preImage is none and estimation failed !!!!!! At least two images are needed for the ego-motion estimation through optical flow
                  please refer to the funciton setPreImg to solve this bug""")
            return
        
        mask = pu.G_cutoff(self.preImg)

        good_old_raw, good_new_raw, flow_raw = pu.cv_featureLK(self.preImg, nextImg, self.deltaT, mask)
        good_old_filter, good_new_filer, flow_filter = em.pre_filter(good_old_raw, good_new_raw, flow_raw, self.h, self.f, self.raw_opVl, self.raw_opW)

        try:
          V_long_raw, w_raw, Error_raw = em.WVReg(good_old_raw, flow_raw, self.h, self.f) # use egomotion estimation function to estimate the ego motion
          V_long_filter, w_filter, Error_filter = em.WVReg(good_old_filter, flow_filter, self.h, self.f) # use egomotion estimation function to estimate the ego motion

          # apply optimization algorithms here
          V_long_final = em.past_fusion(self.raw_opVl, self.raw_Errors, V_long_filter, Error_filter)
          w_final = em.past_fusion(self.raw_opW, self.raw_Errors, w_filter, Error_filter)

          # sometimes there might be some extreme conditions that make regression failed, i.e. almost no flow point can be used for regression at this time an index error will be raised in WVReg
        except IndexError:
            V_long_raw, w_raw, Error_raw = em.past_extreme(self.raw_opVl, self.raw_opW, self.raw_Errors)
            V_long_final, w_final, Error_filter = V_long_raw, w_raw, Error_raw
        
        self.raw_opW.append(w_raw)
        self.raw_opVl.append(V_long_raw)
        self.filter_opW.append(w_final)
        self.filter_opVl.append(V_long_final)
        self.raw_Errors.append(Error_raw)
        self.filter_Errors.append(Error_filter)

        self.preImg = nextImg
        
        return V_long_final, w_final