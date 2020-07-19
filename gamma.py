import numpy as np
import time
import emcee
from multiprocessing import Pool
from binnedFit_utilities import *
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

import sys
sys.path.append("/Users/hhg/Research/kinematic_lensing/code/BinnedFit/")
from GaussFit_spec2D import GaussFit_spec2D
from ImageFit import ImageFit
from RotationCurveFit import RotationCurveFit

class Gamma():

    def __init__(self, data_info, sigma_TF_intr=0.08, gamma_x_mode=True):
        '''

        '''

        self.ImgFit = ImageFit(data_info) # active_par_key = ['e_obs', 'half_light_radius']
        self.RotFit_major = RotationCurveFit(data_info, active_par_key=['vscale', 'r_0', 'v_0', 'v_spec'], data_key="data_major")

        self.active_par_key = self.ImgFit.active_par_key + ['v_TF', 'vscale', 'r_0', 'v_0', 'v_spec_major']
        self.derived_par_key = ['sini', 'e_int', 'gamma_p']

        self.gamma_x_mode = gamma_x_mode
        self.sigma_TF_intr = sigma_TF_intr

        if self.gamma_x_mode==True:
            if "data_minor" in data_info.keys():
                self.RotFit_minor = RotationCurveFit(data_info, active_par_key=['vscale', 'r_0', 'v_0', 'v_spec'], data_key="data_minor")
                
            self.active_par_key += ['v_spec_minor']
            self.derived_par_key += ['gamma_x']
        
        self.par_fid = self.RotFit_major.Parameter.par_fid.copy()
        self.par_std = self.RotFit_major.Parameter.par_std.copy()

    def logPrior_v_TF(self, v_TF):
        '''
            add logPrior on the intrinsic circular velocity of disk based on TF relation
            # this function need to be modified in the future
        '''
        #M_B = -21.8
        # np.log10(np.abs(self.Parameter.par_fid['vcirc']))
        log10_vTFR_mean = np.log10(200.)
        logPrior_v_TF = -0.5 * \
            ((np.log10(np.abs(v_TF)) - log10_vTFR_mean)/self.sigma_TF_intr)**2

        return logPrior_v_TF
    
    def cal_loglike(self, active_par):
        '''
            active_par need to be an array following the order of active_par_key, which is:
            ['e_obs', 'half_light_radius'] + ['v_TF', 'vscale', 'r_0', 'v_0', 'v_spec_major'] + (['v_spec_minor'])
        '''

        active_par_ImgFit = active_par[0:2]
        active_par_Rot_major = active_par[3:7]
        v_TF = active_par[2]
        v_spec_major = active_par[6]
        e_obs = active_par[0]
        #print(active_par_ImgFit)
        #print(active_par_Rot_major)
        #print(v_TF)
        #print(v_spec_major)
        #print(e_obs)

        logL_Image = self.ImgFit.cal_loglike(active_par=active_par_ImgFit)
        logL_Rot_major = self.RotFit_major.cal_loglike(active_par=active_par_Rot_major)

        logPrior_v_TF = self.logPrior_v_TF(v_TF=v_TF)

        # derived parameter
        sini = cal_sini(v_spec=v_spec_major, v_TF=v_TF)
        e_int = cal_e_int(sini=sini, q_z=0.2)
        gamma_p = cal_gamma_p(e_int=e_int, e_obs=e_obs)

        loglike = logL_Image+logL_Rot_major+logPrior_v_TF

        if self.gamma_x_mode == True:
            v_spec_minor = active_par[-1]
            active_par_Rot_minor = np.array(list(active_par[3:6])+[v_spec_minor])
            #print(v_spec_minor)
            #print(active_par_Rot_minor)

            logL_Rot_minor = self.RotFit_minor.cal_loglike(active_par=active_par_Rot_minor)

            gamma_x = cal_gamma_x(v_spec_minor=v_spec_minor, v_TF=v_TF, e_int=e_int, q_z=0.2)

            loglike += logL_Rot_minor

            return loglike, sini, e_int, gamma_p, gamma_x
        
        return loglike, sini, e_int, gamma_p
    
    def run_MCMC(self, Nwalker, Nsteps):

        Ndim = len(self.active_par_key)

        starting_point = [self.par_fid[item] for item in self.active_par_key]
        std = [self.par_std[item] for item in self.active_par_key]

        blobs_dtype = [("sini", float), ("e_int", float), ("gamma_p", float)]

        if self.gamma_x_mode is True:
            blobs_dtype += [("gamma_x", float)]
        
        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)

        sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=2.0, blobs_dtype=blobs_dtype)

        posInfo = sampler.run_mcmc(p0_walkers, 5)
        p0_walkers = posInfo.coords
        sampler.reset()

        Tstart = time.time()
        posInfo = sampler.run_mcmc(p0_walkers, Nsteps, progress=True)
        Time_MCMC = (time.time()-Tstart)/60.
        print ("Total MCMC time (mins):", Time_MCMC)

        chain_sini = np.array(sampler.get_blobs()['sini']).T
        chain_e_int = np.array(sampler.get_blobs()['e_int']).T
        chain_gamma_p = np.array(sampler.get_blobs()['gamma_p']).T

        if self.gamma_x_mode is True:
            chain_gamma_x = np.array(sampler.get_blobs()['gamma_x']).T

        chain_info = {}
        chain_info['acceptance_fraction'] = np.mean(
            sampler.acceptance_fraction)  # good range: 0.2~0.5
        chain_info['lnprobability'] = sampler.lnprobability
        chain_info['par_fid'] = self.par_fid
        chain_info['par_name'] = self.RotFit_major.Parameter.par_name

        if self.gamma_x_mode is True:
            chain_gamma_x = np.array(sampler.get_blobs()['gamma_x']).T
            chain_info['chain'] = np.dstack((np.dstack((np.dstack((np.dstack(
                (sampler.chain[:, ], chain_sini)), chain_e_int)), chain_gamma_p)), chain_gamma_x))
            chain_info['par_key'] = self.active_par_key + self.derived_par_key

        else:
            chain_info['chain'] = np.dstack((np.dstack(
                (np.dstack((sampler.chain[:, ], chain_sini)), chain_e_int)), chain_gamma_p))
            #np.column_stack((sampler.chain[:,], chain_sini, chain_e_int, chain_gamma_p))
            chain_info['par_key'] = self.active_par_key + self.derived_par_key

        return chain_info





        


        

        




        





    



    


        
