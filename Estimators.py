import numpy as np
import data_utils as du
import OP_utils as pu
import Ego_motion as em
import matplotlib.pyplot as plt
import video_utils as vu

class OP_estimator:
    """estimator class for optical flow estimation"""

    def __init__(self, deltaT, h, f, pre_filter_size = 5, pre_filter_discard = 3.0, past_fusion_on = True, past_fusion_size = 10.0, past_fusion_amplifier = 1.5, start_image = None):

        self.f = f
        self.h = h
        self.deltaT = deltaT
        self.preImg = start_image
        self.pre_filter_size = pre_filter_size
        self.pre_filter_discard = pre_filter_discard
        self.past_fusion_on = past_fusion_on
        self.past_fusion_size = past_fusion_size
        self.past_fusion_amplifier = past_fusion_amplifier

        # the recording arrays for the estimation
        self.raw_opW = []
        self.raw_opVl = []
        self.raw_Errors = []
    
    def setPreImg(self, preImg):
        self.preImg = preImg
    
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

        try:
          mask = pu.G_cutoff(preImg) # cut the ground out of the whole image
          good_old, good_new, flow, raw_op_errors = pu.cv_featureLK(preImg, nextImg, deltaT, mask, with_error = True)

          raw_Vl, raw_w, raw_resid = em.WVReg(good_old, flow, h, f) # first regression without any filter
          raw_Error = np.average(raw_op_errors)

          filter_res = em.pre_filter(good_old, good_new, flow, raw_op_errors, h, f, raw_Vls, raw_ws, self.pre_filter_size, self.pre_filter_discard) # use pre measurements without any filtering operation to filter out outliers
          good_old, good_new, flow, op_errors = filter_res[0], filter_res[1], filter_res[2], filter_res[3]
          V_long, w, resid = em.WVReg(good_old, flow, h, f) # use egomotion estimation function to estimate the ego motion
          Error = np.average(op_errors)

          if self.past_fusion_on:
              final_V_long, final_error = em.past_fusion(raw_Vls, raw_Errors, V_long, Error, past_amplifier= self.past_fusion_amplifier, past_steps=self.past_fusion_size)
              final_w, final_error = em.past_fusion(raw_ws, raw_Errors, w, Error)

        # sometimes there might be some extreme conditions that make regression failed, i.e. almost no flow point can be used for regression at this time an index error will be raised in WVReg
        except IndexError:
            final_V_long, final_w, final_error = em.past_extreme(raw_Vls, raw_ws, raw_Errors)
        
        # sometimes even the regression for the first unfiltered regression may fail, which makes raw_Vl, raw_w, and raw_Error not defined
        try:
            raw_Vls.append(raw_Vl)
            raw_ws.append(raw_w)
            raw_Errors.append(raw_Error)

        except UnboundLocalError:
            raw_Vls.append(final_V_long)
            raw_ws.append(final_w)
            raw_Errors.append(final_error)

        self.preImg = nextImg

        return final_V_long, final_w, final_error