import sys
sys.path.append("/Users/hhg/Research/kinematic_lensing/code/BinnedFit/")
from binnedFit_utilities import *
import numpy as np
from Gamma import *


dir_tfCube_data = "/Users/hhg/Research/kinematic_lensing/data/mock_tfCube/"
dir_chain = "/Users/hhg/Research/kinematic_lensing/data/mock_tfCube/chain/"


#g2_list = [0., 0.025, 0.05, 0.075, 0.1]
g2_list = [0.075, 0.1]


for g2 in g2_list:
    sini=0.5
    g1=0.1    ### [0., 0.025, 0.05, 0.075, 0.1]
    fname_read = f"tfCube_sini_{sini:.2f}_g1_{g1:.3f}_g2_{g2:.3f}.pkl"
    data_info=load_pickle(dir_tfCube_data+fname_read)
    
    Gamma_now=Gamma(data_info=data_info, sigma_TF_intr=0.08, mode_gamma_x=True)
    chain_info = Gamma_now.run_MCMC(Nwalker=100, Nsteps=3000)
    
    fname_chain_info = f"chain_sini_{sini:.2f}_g1_{g1:.3f}_g2_{g2:.3f}.pkl"
    
    save_pickle(dir_chain+fname_chain_info, chain_info)
