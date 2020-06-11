import numpy as np
from scipy.optimize import curve_fit
import time
import emcee
from multiprocessing import Pool

from binnedFit_utilities import *
from GaussFit_spec2D import GaussFit_spec2D
import galsim



class ImageFit():
    def __init__(self, data_info, par_fix=None):
        self.image = data_info['image']
        self.variance = data_info['image_variance']
        self.grid_pos = data_info['grid_Image']
        
        self.parameters = data_info['par_fid']

        self.active_par_key = ['e_obs', 'half_light_radius']
        self.Ntot_active_par = len(self.active_par_key)
        self.par_lim = self.def_par_lim()
        self.par_name = {'e_obs': "$e_{\\mathrm obs}$",
                         'half_light_radius': "$r_{\\mathrm 1/2}$"}
        self.par_fid = {'e_obs': 0.3,
                        'half_light_radius': 0.5}
    
    def def_par_lim(self):
        par_lim = {}
        par_lim['e_obs'] = [0., 1.]
        par_lim['half_light_radius'] = [0.0, 2.]
        return par_lim

    def model_image(self, e_obs, half_light_radius):

        psf = galsim.Gaussian(fwhm=self.parameters['psfFWHM'])

        g1_int = e_obs/2

        disk = galsim.Sersic(n=1, half_light_radius=half_light_radius, flux=1)
        disk = disk.shear(g1=g1_int, g2=0.0)

        galObj = galsim.Convolution([disk, psf])
        image = galsim.Image(
            self.parameters['image_size'], self.parameters['image_size'], scale=self.parameters['pixScale'])
        newImage = galObj.drawImage(image=image)

        return newImage.array

    def cal_chi2(self, active_par):
        '''
            active_par : e_obs, half_light_radius 
        '''
        model = self.model_image(
            e_obs=active_par[0], half_light_radius=active_par[1])
        
        diff = self.image - model
        chi2 = np.sum(diff**2 / self.variance)

        return chi2

    def cal_loglike(self, active_par):

        for ind, item in enumerate(self.active_par_key):
            if (active_par[ind] < self.par_lim[item][0] or active_par[ind] > self.par_lim[item][1]):
                return -np.inf
        
        chi2 = self.cal_chi2(active_par)
        loglike = -0.5*chi2

        return loglike
    
    def run_MCMC(self, Nwalker, Nsteps):

        Ndim = self.Ntot_active_par
        starting_point = [0.5, 0.5] # e_obs, half_light_radius
        std = [0.01, 0.01]
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
        chain_info['par_name'] = self.par_name

        return chain_info






    


