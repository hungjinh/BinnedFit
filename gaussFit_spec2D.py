import sys
sys.path.append("../scrips/")
sys.path.append("../repo/KLens/")

import numpy as np
from scipy.optimize import curve_fit
import time

from binnedFit_utilities import *


def load_data(fname_data,fname_grid_lambda):
    data_info = {}
    data_info['data']         = np.load(fname_data)
    data_info['grid_lambda']  = np.load(fname_grid_lambda)
    extent = 4.096
    subGridPixScale = 0.016
    data_info['grid_pos']  = np.arange(-extent/2., extent/2., subGridPixScale)
    return data_info



class gaussFit_spec2D():
    def __init__(self,data_info,lambda0,redshift):
        self.lambda0     = lambda0
        self.redshift    = redshift

        self.data        = data_info['data']
        self.grid_lambda = data_info['grid_lambda']
        self.grid_pos    = data_info['grid_pos']
        self.grid_v      = lambda_to_velocity(lambda0=self.lambda0,lambda_obs=self.grid_lambda,norm=0)

        self.Ngrid_pos, self.Ngrid_spec = self.data.shape

    def _fit_Gauss1D_at_pos(self,data,pos_id,grid_spec,fit_function=gaussian):
        '''
            fit 1D gaussian for data at fixed position (given pos_id :data[pos_id])
            use get_peak_info as an initial starting point before running optimizer
            grid_spec can be either in unit of lambda (grid_lambda as default) or in unit of normalized velocity (grid_v)
        '''

        peak_info = get_peak_info(data,grid_spec)

        if any(x<0 for x in grid_spec):
            init_sigma=100. # if there exists negative elements in grid_spec, meaning the unit is in velocity
                            # set the unit of initial dispersion in orders of sigma_v
        else:
            init_sigma=1.   # if there's no negative element, meaning that the unit of grid_spec is in lambda

        init_vals = [peak_info['peak_loc'][pos_id],peak_info['peak_flux'][pos_id],init_sigma] # for [x0,amp,sigma]
        best_vals,covar = curve_fit(fit_function,grid_spec,data[pos_id],p0=init_vals) # curve_fit(fit_fun,x,f(x),p0=initial_par_values)

        return best_vals

    def fit_spec2D(self,data,grid_spec):
        '''
            loop over each position stripe to get fitted_amp,fitted_peakLoc,fitted_sigma
            fitted_peakLoc unit : same as input grid_spec (velocity or lambda)
        '''
        fitted_amp     = np.zeros(self.Ngrid_pos)
        fitted_peakLoc = np.zeros(self.Ngrid_pos)
        fitted_sigma   = np.zeros(self.Ngrid_pos)

        start_time=time.time()

        for j in range(self.Ngrid_pos):
            fitted_peakLoc[j],fitted_amp[j],fitted_sigma[j] = self._fit_Gauss1D_at_pos(data,j,grid_spec,fit_function=gaussian)

        end_time = time.time()
        print("time cost:",(end_time-start_time),"(secs)")

        return fitted_peakLoc,fitted_amp,fitted_sigma

    def approxFit_rotation_curve(self,peak_lambda):
        '''
            approximately fit arctan velocity curve based on best-fitted peak lambda,
            in order to get an initial guess of arctan velocity profile parameters
        '''
        peak_V = lambda_to_velocity(self.lambda0,peak_lambda) # unit tranformation to total velocity
        init_guess = [0., 0.5, velocity_system(self.redshift), 200.]# for [r_0, r_t, v_0, v_a]
        best_vals, covar = curve_fit(arctan_rotation,self.grid_pos,peak_V,p0=init_guess,bounds=([-50.,0.,0.,-3000.],[50.,50.,6e5,3000.]))

        self.approxFit_rot_v_peak      = arctan_rotation(self.grid_pos,*best_vals)
        self.approxFit_rot_lambda_peak = velocity_to_lambda(self.lambda0,self.approxFit_rot_v_peak)

        params_rot = {}
        for id,par_key_i in enumerate(['r_0','r_t','v_0','v_a']):
            params_rot[par_key_i] = best_vals[id]

        return params_rot

    def gen_model_spec2D(self,fitted_peakLoc,fitted_amp,fitted_sigma,grid_spec):
        '''
            generate model 2D spectrum based on best fitted parameters derived from fit_spec2D
        '''
        model = np.zeros([self.Ngrid_pos,self.Ngrid_spec])

        for j in range(self.Ngrid_pos):
            model[j] = gaussian(grid_spec,fitted_peakLoc[j],fitted_amp[j],fitted_sigma[j])

        return model
