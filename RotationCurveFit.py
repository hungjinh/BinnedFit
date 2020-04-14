import numpy as np
from scipy.optimize import curve_fit
import time
import emcee
from multiprocessing import Pool

from binnedFit_utilities import *
from GaussFit_spec2D import GaussFit_spec2D


class Parameter():
    def __init__(self, par_tfCube, par_fix=None):
        self.par_fid = self.gen_par_fiducial(par_tfCube)
        self.par_fix = par_fix
        self.par_set = self.par_set(par_fix=self.par_fix)

        self.all_par_key = ['r_t', 'r_0', 'v_a', 'v_0', 'redshift', 'sini']
        self.par_absID = {item:j for j, item in enumerate(self.all_par_key)}

        self.par_std = self.def_par_std()
        self.par_lim = self.def_par_lim()
        self.par_name = self.def_par_name()

    def gen_par_fiducial(self, par_tfCube):
        par_fid = {}
        par_fid['sini'] = par_tfCube['sini']
        par_fid['redshift'] = par_tfCube['redshift']
        par_fid['r_0'] = 0.0
        par_fid['r_t'] = par_tfCube['vscale']
        par_fid['v_0'] = 0.0
        par_fid['v_a'] = par_tfCube['vcirc']
        return par_fid

    def def_par_lim(self):
        par_lim = {}
        par_lim['sini'] = [0., 1.]
        par_lim['redshift'] = [self.par_fid['redshift']-0.0035, self.par_fid['redshift']+0.0035]
        par_lim['r_0'] = [-2., 2.]
        par_lim['r_t'] = [0., 1.]
        par_lim['v_0'] = [-1000., 1000.]
        par_lim['v_a'] = [-1000., 1000.]
        return par_lim
    
    def def_par_std(self):
        '''
            define the initial std of emcee walkers around starting point 
        '''
        par_std = {}
        par_std['sini'] = 0.1
        par_std['redshift'] = 0.001
        par_std['r_0'] = 0.1
        par_std['r_t'] = 0.1
        par_std['v_0'] = 20.
        par_std['v_a'] = 20.
        return par_std
    
    def def_par_name(self):
        par_name = {}
        par_name['sini'] = "${\mathrm sin}(i)$"
        par_name['redshift'] = "$z_{\mathrm c}$"
        par_name['r_0'] = "$r_0$"
        par_name['r_t'] = "$r_{\mathrm t}$"
        par_name['v_0'] = "$v_0$"
        par_name['v_a'] = "$v_{\mathrm circ}$"
        return par_name

    def par_set(self, par_fix=None):

        par_set = self.par_fid.copy()
        
        if par_fix is not None:
            for item in list(par_fix.keys()):
                par_set[item] = par_fix[item]

        return par_set

    
    def gen_par_dict(self, active_par, active_par_key):
        par = self.par_set.copy()

        for j,item in enumerate(active_par_key):
            par[item] = active_par[j]
            #par[item] = np.atleast_2d(active_par[:,j])  ####SSS ... use reshape
            # print(par[item].shape)
        return par

    def append_par(self, par_partial):

        par = self.par_set.copy()

        for item in list(par_partial.keys()):
            par[item] = par_partial[item] 
        
        return par


class RotationCurveFit(GaussFit_spec2D):

    def __init__(self, data_info, active_par_key=['r_t', 'r_0', 'v_a', 'v_0', 'redshift'], par_fix=None):
        '''
            e.g. 
            active_par_key = ['r_t', 'r_0', 'v_a', 'v_0']
            par_fix = {'redshift': 0.598}
        '''

        super().__init__(data_info)
        self.Parameter = Parameter(par_tfCube = data_info['par_fid'], par_fix=par_fix)

        self.active_par_key = active_par_key
        self.fix_par_key    = [item for item in self.Parameter.all_par_key if item not in self.active_par_key]
        self.Ntot_active_par = len(self.active_par_key)

        self.gaussfit_peakLambda, self.gaussfit_amp, self.gaussfit_sigma = self.gaussFit_spec2D(data=self.data)


    def model_arctan_rotation(self, r, r_t, r_0, v_a, v_0, redshift, sini):
        '''
            arctan rotation curve in unit of lambda_obs, given cosmological redshift
        '''
        peak_Vp = arctan_rotation_curve(r, r_t, r_0, v_a, v_0, sini)
        model_lambda = velocity_to_lambda(v_peculiar=peak_Vp, lambda_emit=self.lambda_emit, redshift=redshift)

        return model_lambda


    def optFit_rotation_curve(self, fitted_peakLambda, par_init_guess=None):
        '''
            fit an arctan velocity curve based on the best-fit lambda peak (as a function of position grid),
            to get an estimation on the velocity profile parameters
            par_init_guess : {'r_t': 3., 'v_a':200.}
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

        model = self.model_arctan_rotation(self.grid_pos, r_t=par['r_t'], r_0=par['r_0'], v_a=par['v_a'], v_0=par['v_0'], redshift=par['redshift'], sini=par['sini'])
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

        #########SS
        #for j, item in enumerate(self.active_par_key):
        #    # print(par[item].shape)
        #    x= np.logical_or(par[item][0] < self.Parameter.par_lim[item][0], par[item][0] > self.Parameter.par_lim[item][1])
        #    loglike[x]=-np.inf
        #########SS

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
        chain_info['chain'] = sampler.chain
        chain_info['par_key'] = self.active_par_key
        chain_info['par_fid'] = self.Parameter.par_fid
        chain_info['par_name'] = self.Parameter.par_name

        return chain_info






