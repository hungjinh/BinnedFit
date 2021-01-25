import numpy as np
import time
import emcee
from multiprocessing import Pool
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import sys
import pathlib

from binnedFit_utilities import *
from ImageFit import ImageFit
from RotationCurveFit import RotationCurveFit

dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)
from tfCube2 import Parameters

class Gamma():

    def __init__(self, data_info, active_par_key=['vcirc', 'sini', 'vscale', 'r_0', 'v_0', 'g1', 'g2',  'half_light_radius', 'theta_int'], par_fix=None):

        self.sigma_TF_intr = 0.08

        self.Pars = Parameters(par_in=data_info['par_fid'], line_species=data_info['line_species'])

        self.par_fid = data_info['par_fid']
        self.par_fix = par_fix

        if self.par_fix is not None:
            self.par_base = self.Pars.gen_par_dict(active_par=list(self.par_fix.values()), active_par_key=list(self.par_fix.keys()), par_ref=self.par_fid)
        else:
            self.par_base = self.par_fid.copy()
        
        self.ImgFit = ImageFit(data_info, active_par_key=['sini', 'half_light_radius', 'theta_int', 'g1', 'g2'], par_fix=par_fix) 
        self.RotFit = RotationCurveFit(data_info, active_par_key=active_par_key, par_fix=par_fix)

        self.active_par_key_img = self.ImgFit.active_par_key
        self.active_par_key = active_par_key

        self.par_lim = self.Pars.set_par_lim()  # defined in tfCube2.Parameters.set_par_lim()
        self.par_std = self.Pars.set_par_std()

    def cal_loglike(self, active_par):
        '''
        '''

        par = self.Pars.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key, par_ref=self.par_base)

        for item in self.active_par_key:
            if (par[item] < self.par_lim[item][0] or par[item] > self.par_lim[item][1]):
                return -np.inf
        
        logPrior_vcirc = self.Pars.logPrior_vcirc(vcirc=par['vcirc'], sigma_TF_intr=self.sigma_TF_intr)

        active_par_ImgFit = [par[item_key] for item_key in self.active_par_key_img]
        logL_img = self.ImgFit.cal_loglike(active_par=active_par_ImgFit)
        logL_spec = self.RotFit.cal_loglike(active_par=active_par)
        #print('logL_img:', logL_img)
        #print('logL_spec:', logL_spec)

        loglike = logL_img+logL_spec+logPrior_vcirc

        return loglike
    
    def run_MCMC(self, Nwalker, Nsteps):

        Ndim = len(self.active_par_key)

        starting_point = [self.par_fid[item] for item in self.active_par_key]
        std = [self.par_std[item] for item in self.active_par_key]

        
        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)

        sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=2.0)

        posInfo = sampler.run_mcmc(p0_walkers, 5)
        p0_walkers = posInfo.coords
        sampler.reset()

        Tstart = time.time()
        posInfo = sampler.run_mcmc(p0_walkers, Nsteps, progress=True)
        Time_MCMC = (time.time()-Tstart)/60.
        print ("Total MCMC time (mins):", Time_MCMC)


        chain_info = {}
        chain_info['acceptance_fraction'] = np.mean(sampler.acceptance_fraction)  # good range: 0.2~0.5
        chain_info['lnprobability'] = sampler.lnprobability
        chain_info['par_fid'] = self.par_fid
        chain_info['chain'] = sampler.chain
        chain_info['par_key'] = self.active_par_key
        chain_info['par_fix'] = self.par_fix

        return chain_info





        


        

        




        





    



    


        
