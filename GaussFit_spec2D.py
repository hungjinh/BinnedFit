import numpy as np
from scipy.optimize import curve_fit
import time

from binnedFit_utilities import *

class GaussFit_spec2D():
    def __init__(self, data_info):

        self.data        = data_info['data']
        self.lambda_emit = data_info['lambda_emit']
        self.grid_lambda = data_info['grid_lambda']
        self.grid_pos    = data_info['grid_pos']

        self.Ngrid_pos, self.Ngrid_spec = self.data.shape

        self.peak_info_exact = get_peak_info(data=self.data, grid_spec=self.grid_lambda)
        #self.gaussFit_spec2D(data=self.data)
        

    def _fit_Gauss1D_at_pos(self, data, pos_id, fit_function=gaussian):
        '''
            fit 1D gaussian for data at fixed position (given pos_id: data[pos_id])
            use get_peak_info as an initial starting point before running optimizer
        '''

        peak_info = get_peak_info(data, self.grid_lambda)

        init_sigma = 1.     # initial guess on the velocity dispersion at fixed position (here the unit is in nm)
                            # init_sigma need to be in the same unit as self.grid_lambda

        init_vals = [peak_info['peak_loc'][pos_id], peak_info['peak_flux'][pos_id], init_sigma] # for [x0,amp,sigma]
        best_vals, covar = curve_fit(fit_function, self.grid_lambda, data[pos_id], p0=init_vals) # curve_fit(fit_fun,x,f(x),p0=initial_par_values)

        return best_vals

    def gaussFit_spec2D(self, data):
        '''
            loop over each position stripe to get fitted_amp, fitted_peakLambda, fitted_sigma
            fitted_peakLambda unit: same as self.grid_lambda
        '''
        gaussfit_amp     = np.zeros(self.Ngrid_pos)
        gaussfit_peakLambda = np.zeros(self.Ngrid_pos)
        gaussfit_sigma   = np.zeros(self.Ngrid_pos)

        start_time = time.time()

        for j in range(self.Ngrid_pos):
            gaussfit_peakLambda[j], gaussfit_amp[j], gaussfit_sigma[j] = self._fit_Gauss1D_at_pos(data, j, fit_function=gaussian)

        end_time = time.time()
        print("time cost in gaussFit_spec2D:",(end_time-start_time),"(secs)")

        return gaussfit_peakLambda, gaussfit_amp, gaussfit_sigma


    def model_spec2D(self, fitted_peakLambda, fitted_amp, fitted_sigma):
        '''
            generate model 2D spectrum based on best fitted parameters derived from fit_spec2D
        '''
        model_spec2D = np.zeros([self.Ngrid_pos, self.Ngrid_spec])

        for j in range(self.Ngrid_pos):
            model_spec2D[j] = gaussian(self.grid_lambda, fitted_peakLambda[j], fitted_amp[j], fitted_sigma[j])

        return model_spec2D
    

