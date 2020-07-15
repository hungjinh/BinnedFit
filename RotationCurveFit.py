import numpy as np
from scipy.optimize import curve_fit
import time
import emcee
from multiprocessing import Pool

from binnedFit_utilities import *
from GaussFit_spec2D import GaussFit_spec2D


class RotationCurveFit(GaussFit_spec2D):

    def __init__(self, data_info, e_obs=None, active_par_key=['vscale', 'r_0', 'vcirc', 'v_0'], par_fix=None, sigma_TF_intr=0.05):
        '''
            e.g. 
            active_par_key = ['vscale', 'r_0', 'vcirc', 'v_0']
            par_fix = {'redshift': 0.598}
        '''

        super().__init__(data_info)
        self.Parameter = Parameter(par_tfCube = data_info['par_fid'], par_fix=par_fix)
        self.sigma_TF_intr = sigma_TF_intr

        self.active_par_key = active_par_key
        self.fix_par_key    = [item for item in self.Parameter.all_par_key if item not in self.active_par_key]
        self.Ntot_active_par = len(self.active_par_key)
        
        self.remove_0signal_grid()
        

        if e_obs is not None:
            self.shear_mode = 1
            self.e_obs = e_obs
        else:
            self.shear_mode = 0

    def remove_0signal_grid(self, threshold_amp=1e-20):
        '''
            check positions where the peak amp is small, and remove these position info.
        '''

        self.gaussfit_peakLambda, self.gaussfit_amp, self.gaussfit_sigma = self.gaussFit_spec2D(data=self.data)
        
        ID_small_amp = np.where(np.abs(self.gaussfit_amp) < threshold_amp)[0]

        if len(ID_small_amp) != 0 :
            self.ID_keep = np.where(np.abs(self.gaussfit_amp) >= threshold_amp)[0]

            self.gaussfit_amp = self.gaussfit_amp[self.ID_keep]
            self.gaussfit_peakLambda = self.gaussfit_peakLambda[self.ID_keep]
            self.gaussfit_sigma = self.gaussfit_sigma[self.ID_keep]
            self.grid_pos = self.grid_pos[self.ID_keep]


    def model_arctan_rotation(self, r, vscale, r_0, vcirc, v_0, redshift, sini):
        '''
            arctan rotation curve in unit of lambda_obs, given cosmological redshift
        '''
        peak_Vp = arctan_rotation_curve(r, vscale, r_0, vcirc, v_0, sini)
        model_lambda = velocity_to_lambda(v_peculiar=peak_Vp, lambda_emit=self.lambda_emit, redshift=redshift)

        return model_lambda


    def optFit_rotation_curve(self, fitted_peakLambda, par_init_guess=None):
        '''
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
        
        if 'cosi' in self.active_par_key:
            par['sini'] = np.sqrt(1-par['cosi']**2)

        model = self.model_arctan_rotation(self.grid_pos, vscale=par['vscale'], r_0=par['r_0'], vcirc=par['vcirc'], v_0=par['v_0'], redshift=par['redshift'], sini=par['sini'])
        #print(model_1D_lambdaPeak)

        #diff = self.gaussfit_peakLambda.reshape(256,1) - model
        #chi2 = np.sum((diff/self.gaussfit_sigma.reshape(256,1))**2,axis=0)

        diff = self.gaussfit_peakLambda - model
        chi2 = np.sum((diff/self.gaussfit_sigma)**2)

        return chi2, par['sini']

    def cal_loglike(self, active_par):
        
        par = self.Parameter.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key)

        for item in self.active_par_key:
            if ( par[item] < self.Parameter.par_lim[item][0] or par[item] > self.Parameter.par_lim[item][1] ):
                return -np.inf, -99., -99.
        ######### vectorized #########
        #for j, item in enumerate(self.active_par_key):
        #    # print(par[item].shape)
        #    x= np.logical_or(par[item][0] < self.Parameter.par_lim[item][0], par[item][0] > self.Parameter.par_lim[item][1])
        #    loglike[x]=-np.inf
        ######### vectorized #########

        chi2, sini = self.cal_chi2(active_par)

        ### informative prior ###
        if 'vcirc' in self.active_par_key:
            logPrior_vcirc = self.logPrior_vcirc(vcirc=par['vcirc'])
        else: 
            logPrior_vcirc = 0
        ### informative prior ###

        loglike = -0.5*chi2 + logPrior_vcirc

        e_int = self.cal_e_int(sini=sini, aspect=self.Parameter.par_fid['aspect'])
        g1 = self.cal_gamma1(e_int=e_int, e_obs=self.e_obs)

        return loglike, e_int, g1

    
    def logPrior_vcirc(self, vcirc):
        '''
            add logPrior on intrinsic circular velocity of disk based on TF relation
        '''
        #M_B = -21.8
        log10_vTFR_mean = np.log10(np.abs(self.Parameter.par_fid['vcirc'])) # 2.142-0.128*(M_B + 20.558)
        logPrior_vcirc = -0.5 * \
            ((np.log10(np.abs(vcirc)) - log10_vTFR_mean)/self.sigma_TF_intr)**2

        #vTFR_mean = np.abs(self.Parameter.par_fid['vcirc'])
        #logPrior_vcirc = -0.5 * ((np.abs(vcirc) - vTFR_mean)/self.sigma_TF_intr)**2

        return logPrior_vcirc
    
    def cal_e_int(self, sini, aspect):
        return ( (1-aspect**2)*sini**2 ) / ( 2-(1-aspect**2)*sini**2 )
    
    def cal_gamma1(self, e_int, e_obs):
        return (e_obs-e_int)/2/(1-e_int**2)

    def run_MCMC(self, Nwalker, Nsteps):

        Ndim = self.Ntot_active_par
        starting_point = [ self.Parameter.par_fid[item] for item in self.active_par_key ]
        std = [ self.Parameter.par_std[item] for item in self.active_par_key ]

        blobs_dtype = [("e_int", float), ("g1", float)]
        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)

        sampler = emcee.EnsembleSampler(
            Nwalker, Ndim, self.cal_loglike, a=2.0, blobs_dtype=blobs_dtype)
        # emcee a parameter: Npar < 4 -> better set a > 3  ( a = 5.0 )
                    #                  : Npar > 7 -> better set a < 2  ( a = 1.5 ) 
        posInfo = sampler.run_mcmc(p0_walkers,5)
        p0_walkers = posInfo.coords
        sampler.reset()
            
        Tstart=time.time()
        posInfo = sampler.run_mcmc(p0_walkers, Nsteps, progress=True)
        Time_MCMC=(time.time()-Tstart)/60.
        print ("Total MCMC time (mins):",Time_MCMC)

        chain_e_int = np.array(sampler.get_blobs()['e_int']).T
        chain_gamma = np.array(sampler.get_blobs()['g1']).T

        chain_info = {}
        chain_info['acceptance_fraction'] = np.mean(sampler.acceptance_fraction) # good range: 0.2~0.5
        chain_info['lnprobability'] = sampler.lnprobability
        chain_info['par_fid'] = self.Parameter.par_fid
        chain_info['par_name'] = self.Parameter.par_name

        chain_info['chain'] = np.dstack( (np.dstack((sampler.chain[:,], chain_e_int)), chain_gamma))
        chain_info['par_key'] = self.active_par_key + ['e_int', 'g1']

        return chain_info






