import numpy as np
from scipy.optimize import curve_fit
import time
import emcee
from multiprocessing import Pool

import galsim
from binnedFit_utilities import *

import sys
import pathlib
dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)
from tfCube2 import Parameters, GalaxyImage


class ImageFit(GalaxyImage):
    def __init__(self, data_info, active_par_key=['sini', 'half_light_radius', 'theta_int', 'g1', 'g2'], par_fix=None):

        self.par_fid = data_info['par_fid']
        self.par_fix = par_fix

        if self.par_fix is not None:
            self.par_base = self.gen_par_dict(active_par=list(self.par_fix.values()), active_par_key=list(self.par_fix.keys()), par_ref=self.par_fid)
        else:
            self.par_base = self.par_fid.copy()
        
        super().__init__(pars=self.par_base, flux_norm=data_info['flux_norm'])

        self.image = data_info['image']
        self.variance = data_info['image_variance']
        
        self.active_par_key = active_par_key
        self.Ntot_active_par = len(self.active_par_key)
        
        self.par_lim = self.set_par_lim()
        self.par_std = self.set_par_std()

    def cal_chi2(self, active_par):
        '''
            active_par : half_light_radius, theta_int, ...
        '''
        par = self.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key, par_ref=self.par_base)

        e = cal_e_int(sini=par['sini'], q_z=par['aspect'])
        model = self.model(e=e, half_light_radius=par['half_light_radius'], theta_int=par['theta_int'], g1=par['g1'], g2=par['g2'])
        
        diff = self.image - model
        chi2 = np.sum(diff**2 / self.variance)

        return chi2

    def cal_loglike(self, active_par):

        par = self.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key, par_ref=self.par_base)

        for ind, item in enumerate(self.active_par_key):
            if (par[item] < self.par_lim[item][0] or par[item] > self.par_lim[item][1]):
                return -np.inf
        
        chi2 = self.cal_chi2(active_par)
        loglike = -0.5*chi2

        return loglike
    
    def run_MCMC(self, Nwalker, Nsteps):

        Ndim = self.Ntot_active_par
        starting_point = [self.par_fid[item] for item in self.active_par_key]
        std = [self.par_std[item] for item in self.active_par_key]
        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)

        sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=5.0)
        posInfo = sampler.run_mcmc(p0_walkers, 5)
        p0_walkers = posInfo.coords
        sampler.reset()

        Tstart = time.time()
        posInfo = sampler.run_mcmc(p0_walkers, Nsteps, progress=True)
        Time_MCMC = (time.time()-Tstart)/60.
        print("Total MCMC time (mins):", Time_MCMC)

        chain_info = {}
        chain_info['acceptance_fraction'] = np.mean(
            sampler.acceptance_fraction)  # good range: 0.2~0.5
        chain_info['lnprobability'] = sampler.lnprobability
        chain_info['chain'] = sampler.chain
        chain_info['par_key'] = self.active_par_key
        chain_info['par_fid'] = self.par_fid
        chain_info['par_fix'] = self.par_fix

        return chain_info






    


