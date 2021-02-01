import numpy as np
from chainconsumer import ChainConsumer

import sys
import pathlib
dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)
from tfCube2 import Parameters

blue        = "#214F94"
red         = "#CC0204"
yellow      = "#FFA903"
gray        = "#858278"
black       = "#000000"
green       = "#73ab24"
purple      = "#8600C8"
lightblue   = "#6898FF"
lightblue2  = "#82b9e6"
yellowgreen = "#8dcd32" #"#8DE900"
lightteal   = "#7FC2BF"
teal        = "#51ABAE"
lightgray   = "#CDCDCD"


class ChainTool(Parameters):
    def __init__(self, chain_info, Nburn, Nend=None):
        
        self.raw_chain = chain_info['chain']
        self.Nwalker, self.iteration, self.dim = self.raw_chain.shape

        exNburn = int(Nburn/self.Nwalker)

        if Nend is not None:
            exNend = int(Nend/self.Nwalker)
        else:
            exNend = self.iteration

        self.chain = self.raw_chain[:, exNburn:exNend, :].reshape((-1, self.dim))
        self.lnprob = chain_info['lnprobability'][:, exNburn:exNend].reshape((-1))

        self.chain_par_key = chain_info['par_key']
        self.par_fid = chain_info['par_fid']
        self.par_name = self.set_par_name()
        
        self.chain_par_id = {item:j for j, item in enumerate(self.chain_par_key)}


    def select_par_info(self, select_par_key):

        select_par_id = [self.chain_par_id[item] for item in select_par_key]
        select_par_name = [self.par_name[item] for item in select_par_key]
        select_par_fid = [self.par_fid[item] for item in select_par_key]
        
        return select_par_id, select_par_name, select_par_fid
    
    def par_bestfit(self, select_par_key=None, statistics='max', mode=0):
        '''
            statistics = 'max' or 'cumulative'
            mode = 0 : return in ChainConsumer summary form
            mode = 1 : return par_bestfit, ave_error = (upper_err+lower_err)/2.
            mode = 2 : return par_bestfit, lower_err, upper_err
        '''
        if select_par_key is None:
            select_par_key = self.chain_par_key

        ndim = len(select_par_key)

        select_par_id, select_par_name, select_par_fid = self.select_par_info(select_par_key)

        c = ChainConsumer()
        c.add_chain(self.chain[:, select_par_id] , parameters=select_par_key)
        c.configure(statistics=statistics)
        summary=c.analysis.get_summary()

        if mode == 0:
            return summary
        else:
            select_par_bestfit = np.array([summary[item][1] for item in select_par_key])
            select_par_lower_err = select_par_bestfit - np.array([summary[item][0] for item in select_par_key])
            select_par_upper_err = np.array([summary[item][2] for item in select_par_key]) - select_par_bestfit
            select_par_avgerr = (select_par_upper_err + select_par_lower_err)/2.

            if mode == 1:
                return select_par_bestfit, select_par_avgerr
            if mode == 2:
                return select_par_bestfit, select_par_lower_err, select_par_upper_err
    
    def par_maxlike(self, chain=None, lnprob=None, select_par_key=None):
        
        if chain is None:
            chain = self.chain
        if lnprob is None:
            lnprob = self.lnprob
        if select_par_key is None:
            select_par_key=self.chain_par_key

        ind_maxlike = np.argmax(lnprob) # max likelihood index
        maxlike_value = lnprob[ind_maxlike]
        par_maxlike = chain[ind_maxlike]

        select_par_id = [self.chain_par_id[item] for item in select_par_key]
        
        return par_maxlike[select_par_id], maxlike_value
    
    def extract_subchain(self, par_bound):
        '''
            extract a subspace of the chain given parameter boundary
            e.g. par_bound={'sini':[0.,0.1], 'g1':[0.05, 0.1]}
        '''

        subchain = self.chain.copy()
        sublnprob = self.lnprob.copy()

        for item in par_bound.keys():

            par_id = self.chain_par_id[item]
            take_out_id = np.where((subchain[:,par_id] > par_bound[item][0]) & (subchain[:,par_id] < par_bound[item][1]))

            subchain = subchain[take_out_id[0], :]
            sublnprob = sublnprob[take_out_id[0]]
        
        return subchain, sublnprob




