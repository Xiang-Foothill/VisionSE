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
        if self.preImg is None:
            print("""Error: the preImage is none and estimation failed !!!!!! At least two images are needed for the ego-motion estimation through optical flow
                  please refer to the funciton setPreImg to solve this bug""")
            return
        
        mask = pu.G_cutoff(self.preImg)

        good_old_raw, good_new_raw, flow_raw = pu.cv_featureLK(self.preImg, nextImg, self.deltaT, mask)
        good_old_filter, good_new_filer, flow_filter = em.pre_filter(good_old_raw, good_new_raw, flow_raw, self.h, self.f, self.filter_opVl, self.filter_opW)
        print(good_old_filter.shape)
        try:
        #   V_long_raw, w_raw, Error_raw = em.WVReg(good_old_raw, flow_raw, self.h, self.f) # use egomotion estimation function to estimate the ego motion
          V_long_filter, w_filter, Error_filter = em.WVReg(good_old_filter, flow_filter, self.h, self.f) # use egomotion estimation function to estimate the ego motion

          # apply optimization algorithms here
          V_long_final = em.past_fusion(self.filter_opVl, self.filter_Errors, V_long_filter, Error_filter)
          w_final = em.past_fusion(self.filter_opW, self.filter_Errors, w_filter, Error_filter)

          # sometimes there might be some extreme conditions that make regression failed, i.e. almost no flow point can be used for regression at this time an index error will be raised in WVReg
        except IndexError:
            V_long_raw, w_raw, Error_raw = em.past_extreme(self.filter_opVl, self.filter_opW, self.filter_Errors)
            V_long_final, w_final, Error_filter = V_long_raw, w_raw, Error_raw
        
        # self.raw_opW.append(w_raw)
        # self.raw_opVl.append(V_long_raw)
        self.filter_opW.append(w_final)
        self.filter_opVl.append(V_long_final)
        # self.raw_Errors.append(Error_raw)
        self.filter_Errors.append(Error_filter)

        self.preImg = nextImg
        
        return V_long_final, w_final
    
    def estimate_copy(self, nextImg):
        preImg = self.preImg
        deltaT = self.deltaT
        h = self.h
        f = self.f
        op_Vl = self.raw_opVl
        op_w = self.raw_opW
        Errors = self.filter_Errors
        est_Vl = self.filter_opVl
        est_w = self.filter_opW

        mask = pu.G_cutoff(preImg)
        good_old, good_new, flow = pu.cv_featureLK(preImg, nextImg, deltaT, mask)

        good_old, good_new, flow = em.pre_filter(good_old, good_new, flow, h, f, op_Vl, op_w)

        print(f"after pre_filter: {good_old.shape[0]}")

        try:
          V_long, w, Error = em.WVReg(good_old, flow, h, f) # use egomotion estimation function to estimate the ego motion
          # apply optimization algorithms here
          final_V_long = V_long
          final_w = em.past_fusion(op_w, Errors, w, Error)

          # sometimes there might be some extreme conditions that make regression failed, i.e. almost no flow point can be used for regression at this time an index error will be raised in WVReg
        except IndexError:
            V_long, w, Error = em.past_extreme(op_Vl, op_w, Errors)
            final_V_long, final_w = V_long, w
        
        op_Vl.append(V_long)
        op_w.append(w)
        est_Vl.append(final_V_long)
        est_w.append(final_w)
        Errors.append(Error)

        self.preImg = nextImg

        return final_V_long, final_w
    
    def estimate_dev(self, nextImg):
        preImg = self.preImg
        deltaT = self.deltaT
        h = self.h
        f = self.f
        raw_Vls = self.raw_opVl
        raw_ws = self.raw_opW
        raw_Errors = self.raw_Errors
        est_Errors = self.filter_Errors
        est_Vls = self.filter_opVl
        est_ws = self.filter_opW

        mask = pu.G_cutoff(preImg)
        good_old, good_new, flow = pu.cv_featureLK(preImg, nextImg, deltaT, mask)
        raw_Vl, raw_w, raw_Error = em.WVReg(good_old, flow, h, f)

        good_old, good_new, flow = em.pre_filter(good_old, good_new, flow, h, f, raw_Vls, raw_ws)
        
        raw_Vls.append(raw_Vl)
        raw_ws.append(raw_w)
        raw_Errors.append(raw_Error)
        
        print(f"after pre_filter: {good_old.shape[0]}")

        try:
          V_long, w, Error = em.WVReg(good_old, flow, h, f) # use egomotion estimation function to estimate the ego motion
          # apply optimization algorithms here
          final_V_long = em.past_fusion(raw_Vls, raw_Errors, V_long, Error)
          final_w = em.past_fusion(raw_ws, raw_Errors, w, Error)

          # sometimes there might be some extreme conditions that make regression failed, i.e. almost no flow point can be used for regression at this time an index error will be raised in WVReg
        except IndexError:
            V_long, w, Error = em.past_extreme(raw_Vls, raw_ws, raw_Errors)
            final_V_long, final_w = V_long, w
        
        est_Vls.append(final_V_long)
        est_ws.append(final_w)
        est_Errors.append(Error)

        self.preImg = nextImg

        return final_V_long, final_w