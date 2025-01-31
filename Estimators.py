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
    
    def estimate(self, nextImg):
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
          raw_Error = np.average(raw_op_errors) # calculate the error of estimation without prefiltering

          filter_res = em.pre_filter(good_old, good_new, flow, raw_op_errors, h, f, raw_Vls, raw_ws, self.pre_filter_size, self.pre_filter_discard) # use pre measurements without any filtering operation to filter out outliers
          good_old, good_new, flow, op_errors = filter_res[0], filter_res[1], filter_res[2], filter_res[3]
          V_long, w, resid = em.WVReg(good_old, flow, h, f) # use egomotion estimation function to estimate the ego motion
          Error = np.average(op_errors) # calculate the error of estimation with filtering

          if self.past_fusion_on: # if the user chooses to apply fast fusion, call the fast fusion function
              final_V_long, final_w, final_error = em.past_fusion(raw_Vls, raw_ws, raw_Errors, V_long, w, Error, past_amplifier= self.past_fusion_amplifier, past_steps=self.past_fusion_size)
          else:
              final_V_long, final_w, final_error = V_long, w, Error

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

        # to optimize space use, only keep the latest measurements of raw_Vls, raw_ws, and raw_Errors
        keep_size = int(max(self.past_fusion_size, self.pre_filter_size)) + 5
        if len(raw_Vls) > keep_size:
            raw_Vls, raw_ws, raw_Errors = raw_Vls[- keep_size : ], raw_ws[- keep_size : ], raw_Errors[- keep_size : ]
            
        return final_V_long, final_w, final_error