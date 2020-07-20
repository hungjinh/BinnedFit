import numpy as np
from scipy.optimize import curve_fit
import time
import emcee
from multiprocessing import Pool

from binnedFit_utilities import *
from GaussFit_spec2D import GaussFit_spec2D


class RotationCurveFit(GaussFit_spec2D):

    def __init__(self, data_info, active_par_key=['vscale', 'r_0', 'v_spec', 'v_0'], par_fix=None, data_key=None):
        '''
            data_key : specify which 2D spec data to use to put in data_info['data']
                       "data_major" or "data_minor"
            e.g. 
            active_par_key = ['vscale', 'r_0', 'v_spec', 'v_0'] # 'redshift'
            par_fix = {'redshift': 0.598}
        '''

        if data_key is not None:
            data_info['data'] = data_info[data_key]

        super().__init__(data_info)
        self.Parameter = Parameter(par_tfCube = data_info['par_fid'], par_fix=par_fix)

        self.active_par_key = active_par_key
        self.fix_par_key    = [item for item in self.Parameter.all_par_key if item not in self.active_par_key]
        self.Ntot_active_par = len(self.active_par_key)
        
        self.remove_0signal_grid()
    

    def remove_0signal_grid(self, threshold_amp=1e-20):
        '''
            check positions where the peak amp is small, and remove these position info.
        '''

        self.gaussfit_peakLambda, self.gaussfit_amp, self.gaussfit_sigma = self.gaussFit_spec2D(data=self.data)
        
        ID_small_amp = np.where(np.abs(self.gaussfit_amp) < threshold_amp)[0]

        self.grid_pos_ori = self.grid_pos.copy()
        if len(ID_small_amp) != 0 :
            self.ID_keep = np.where(np.abs(self.gaussfit_amp) >= threshold_amp)[0]

            self.gaussfit_amp = self.gaussfit_amp[self.ID_keep]
            self.gaussfit_peakLambda = self.gaussfit_peakLambda[self.ID_keep]
            self.gaussfit_sigma = self.gaussfit_sigma[self.ID_keep]
            self.grid_pos = self.grid_pos_ori[self.ID_keep]


    def model_arctan_rotation(self, r, vscale, r_0, v_spec, v_0, redshift):
        '''
            arctan rotation curve in unit of lambda_obs, given cosmological redshift
        '''
        peak_Vp = arctan_rotation_curve(r, vscale, r_0, v_spec, v_0)
        model_lambda = velocity_to_lambda(v_peculiar=peak_Vp, lambda_emit=self.lambda_emit, redshift=redshift)

        return model_lambda


    def optFit_rotation_curve(self, fitted_peakLambda, par_init_guess=None):
        '''
            This code is not compatable within this Class now ... need to be fixed. -- 7/15 2020
            fit an arctan velocity curve based on the best-fit lambda peak (as a function of position grid),
            to get an estimation on the velocity profile parameters
            par_init_guess : {'vscale': 3., 'vcirc':200.}
        '''

        if par_init_guess is not None:
            parFull_init_guess = self.Parameter.append_par(par_init_guess)
            init_guess = [parFull_init_guess[item] for item in self.Parameter.all_par_key]
        else:
            init_guess = [self.Parameter.par_set[item] for item in self.Parameter.all_par_key]

        bound_lower = [self.Parameter.par_lim[item][0] for item in self.Parameter.all_par_key]
        bound_upper = [self.Parameter.par_lim[item][1] for item in self.Parameter.all_par_key]

        # set the upper and lower bounds to be at par_set for parameters that are not allowed to vary (self.fix_par_key)
        for item in self.fix_par_key:
            absID = self.Parameter.par_absID[item]
            bound_upper[absID] = self.Parameter.par_set[item]
            bound_lower[absID] = self.Parameter.par_set[item]-1e-8
        #print(bound_upper)
        #print(bound_lower)

        best_vals, covar = curve_fit(self.model_arctan_rotation, self.grid_pos, fitted_peakLambda, p0=init_guess, bounds=(bound_lower, bound_upper))

        self.fitted_rot_lambdaObs = self.model_arctan_rotation(self.grid_pos, *best_vals)

        params_rot = {item:best_vals[j] for j, item in enumerate(self.Parameter.all_par_key)}

        return params_rot
    
    def cal_chi2(self, active_par):

        par = self.Parameter.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key)
        #print(par)

        model = self.model_arctan_rotation(self.grid_pos, vscale=par['vscale'], r_0=par['r_0'], v_spec=par['v_spec'], v_0=par['v_0'], redshift=par['redshift'])
        #print(model_1D_lambdaPeak)

        #diff = self.gaussfit_peakLambda.reshape(256,1) - model
        #chi2 = np.sum((diff/self.gaussfit_sigma.reshape(256,1))**2,axis=0)

        diff = self.gaussfit_peakLambda - model
        chi2 = np.sum((diff/self.gaussfit_sigma)**2)

        return chi2

    def cal_loglike(self, active_par):
        
        par = self.Parameter.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key)

        for item in self.active_par_key:
            if ( par[item] < self.Parameter.par_lim[item][0] or par[item] > self.Parameter.par_lim[item][1] ):
                return -np.inf
                
        ######### vectorized #########
        #for j, item in enumerate(self.active_par_key):
        #    # print(par[item].shape)
        #    x= np.logical_or(par[item][0] < self.Parameter.par_lim[item][0], par[item][0] > self.Parameter.par_lim[item][1])
        #    loglike[x]=-np.inf
        ######### vectorized #########

        chi2 = self.cal_chi2(active_par)

        loglike = -0.5*chi2

        return loglike

    def run_MCMC(self, Nwalker, Nsteps):

        Ndim = self.Ntot_active_par
        starting_point = [ self.Parameter.par_fid[item] for item in self.active_par_key ]
        std = [ self.Parameter.par_std[item] for item in self.active_par_key ]

        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)

        sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=2.0)
        # emcee a parameter: Npar < 4 -> better set a > 3  ( a = 5.0 )
                    #                  : Npar > 7 -> better set a < 2  ( a = 1.5 ) 
        posInfo = sampler.run_mcmc(p0_walkers,5)
        p0_walkers = posInfo.coords
        sampler.reset()
            
        Tstart=time.time()
        posInfo = sampler.run_mcmc(p0_walkers, Nsteps, progress=True)
        Time_MCMC=(time.time()-Tstart)/60.
        print ("Total MCMC time (mins):",Time_MCMC)

        chain_info = {}
        chain_info['acceptance_fraction'] = np.mean(sampler.acceptance_fraction) # good range: 0.2~0.5
        chain_info['lnprobability'] = sampler.lnprobability
        chain_info['par_fid'] = self.Parameter.par_fid
        chain_info['par_name'] = self.Parameter.par_name
        chain_info['chain'] = sampler.chain
        chain_info['par_key'] = self.active_par_key

        return chain_info

